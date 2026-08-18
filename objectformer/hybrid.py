
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .objects import PseudoInstance, proposal_nodes
from .prompts import nearest_instance_id
from .prototypes import resolve_class_prototypes


def _shifted(image, dx, dy, fill=(124,116,104)):
    arr=np.asarray(image.convert("RGB")); h,w=arr.shape[:2]; out=np.empty_like(arr); out[:]=fill
    sx0,sx1=max(0,-dx),min(w,w-dx); sy0,sy1=max(0,-dy),min(h,h-dy)
    dx0,dx1=max(0,dx),min(w,w+dx); dy0,dy1=max(0,dy),min(h,h+dy)
    if sx1>sx0 and sy1>sy0: out[dy0:dy1,dx0:dx1]=arr[sy0:sy1,sx0:sx1]
    return Image.fromarray(out)


@torch.inference_mode()
def rotation_consistent_feature(encoder,image,cache_path=None):
    if cache_path and os.path.exists(cache_path):
        try:
            return F.normalize(torch.load(cache_path, map_location=encoder.device, weights_only=True).float(), dim=1)
        except (OSError, RuntimeError, EOFError, ValueError):
            # Interrupted concurrent runs may leave a partial cache file.
            # Recompute it and atomically replace it below.
            pass
    base=encoder.extract(image); arr=np.asarray(image.convert("RGB")); feats=[base]
    for k in (1,2,3):
        rotated=Image.fromarray(np.ascontiguousarray(np.rot90(arr,k=k)))
        feats.append(torch.rot90(encoder.extract(rotated),k=-k,dims=(-2,-1)))
    feature = F.normalize(torch.stack(feats).mean(0),dim=1)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp = f"{cache_path}.{os.getpid()}.tmp"
        torch.save(feature.detach().cpu().half(), tmp)
        os.replace(tmp, cache_path)
    return feature


@torch.inference_mode()
def rotation_consistent_features(encoder, images, cache_paths=None):
    """Batch DINO extraction while retaining image-wise rotation fusion/cache."""
    cache_paths = list(cache_paths or [None] * len(images))
    if len(cache_paths) != len(images):
        raise ValueError("cache_paths and images must have the same length")
    results = [None] * len(images)
    missing = []
    for index, path in enumerate(cache_paths):
        if path and os.path.exists(path):
            try:
                results[index] = F.normalize(
                    torch.load(path, map_location=encoder.device, weights_only=True).float(), dim=1,
                )
            except (OSError, RuntimeError, EOFError, ValueError):
                missing.append(index)
        else:
            missing.append(index)
    if missing:
        arrays = [np.asarray(images[index].convert("RGB")) for index in missing]
        fused = [[] for _ in missing]
        for k in (0, 1, 2, 3):
            oriented = [
                images[index] if k == 0 else Image.fromarray(np.ascontiguousarray(np.rot90(array, k=k)))
                for index, array in zip(missing, arrays)
            ]
            batch_features = encoder.extract_batch(oriented)
            if k:
                batch_features = torch.rot90(batch_features, k=-k, dims=(-2, -1))
            for local_index, feature in enumerate(batch_features.split(1, dim=0)):
                fused[local_index].append(feature)
        for local_index, image_index in enumerate(missing):
            feature = F.normalize(torch.stack(fused[local_index]).mean(0), dim=1)
            results[image_index] = feature
            path = cache_paths[image_index]
            if path:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tmp = f"{path}.{os.getpid()}.tmp"
                torch.save(feature.detach().cpu().half(), tmp)
                os.replace(tmp, path)
    return results


def point_patch_embedding(feature,x,y,image_size,side):
    _,_,h,w=feature.shape
    fx=int(round(float(x)/max(image_size[0]-1,1)*(w-1))); fy=int(round(float(y)/max(image_size[1]-1,1)*(h-1)))
    x0=max(0,min(fx-(side-1)//2,w-side)); y0=max(0,min(fy-(side-1)//2,h-side))
    return F.normalize(feature[0,:,y0:y0+side,x0:x0+side].mean((-2,-1)),dim=0)


def adaptive_anchor_prototypes(feature,nodes,instance_map,points,image_size,cfg):
    hcfg=cfg.get("hybrid_anchor",{}); default_side=int(hcfg.get("default_patch_tokens",2))
    patch_sides={int(k): int(v) for k,v in hcfg.get("patch_tokens_by_class",{}).items()}
    car_id=int(hcfg.get("car_class_id",4)); car_side=int(hcfg.get("car_patch_tokens",1))
    def patch_side(cid):
        return patch_sides.get(int(cid), car_side if int(cid)==car_id else default_side)
    sam_weight=float(hcfg.get("sam_weight",0.7)); patch_weight=1.0-sam_weight
    by_id={int(n["mask_id"]):n for n in nodes}; inst=np.asarray(instance_map)
    if not bool(cfg.get("supervision", {}).get("sam_anchor_enabled", True)):
        # DINO-semantic ablation: point patches are the only prototype source.
        # SAM proposals may still be loaded from the shared cache for a common
        # data path, but neither their masks nor embeddings enter supervision.
        vectors={}; patch_only=set()
        for cid,class_points in points.items():
            values=[]
            for x,y in class_points:
                side=patch_side(cid)
                values.append(point_patch_embedding(feature,x,y,image_size,side))
            if values:
                vectors[int(cid)]=F.normalize(torch.stack(values).mean(0),dim=0)
                patch_only.add(int(cid))
        return vectors,{},patch_only
    max_points = max((len(class_points) for class_points in points.values()), default=0)
    if max_points <= 1 and bool(hcfg.get("legacy_matching",True)):
        # Keep the historical one-point path unchanged.  The multi-point path
        # below deliberately adds uniqueness and conflict handling.
        vectors={}; direct_anchors={}; patch_only=set()
        for cid,class_points in points.items():
            values=[]
            for x,y in class_points:
                side=patch_side(cid)
                patch=point_patch_embedding(feature,x,y,image_size,side)
                xi=int(np.clip(round(x),0,inst.shape[1]-1)); yi=int(np.clip(round(y),0,inst.shape[0]-1)); mid=int(inst[yi,xi])
                if mid>0 and mid in by_id:
                    vec=F.normalize(sam_weight*by_id[mid]["embedding"]+patch_weight*patch,dim=0)
                    direct_anchors[mid]=int(cid)
                else:
                    vec=patch; patch_only.add(int(cid))
                values.append(vec)
            if values: vectors[int(cid)]=F.normalize(torch.stack(values).mean(0),dim=0)
        return vectors,direct_anchors,patch_only

    if max_points > 1:
        # Multi-point annotations are instance evidence.  One proposal can be
        # supervised by at most one class and contributes once per class,
        # preventing duplicate points from overweighting a single SAM mask.
        radius=int(cfg.get("prompt",{}).get("point_match_radius",20))
        matched=[]; classes_by_mask={}
        for cid,class_points in points.items():
            for x,y in class_points:
                side=patch_side(cid)
                patch=point_patch_embedding(feature,x,y,image_size,side)
                mid=nearest_instance_id(inst,x,y,radius=radius)
                matched.append((int(cid),patch,int(mid)))
                if mid>0 and mid in by_id:
                    classes_by_mask.setdefault(int(mid),set()).add(int(cid))

        vectors={}; direct_anchors={}; patch_only=set(); values_by_class={}; used_by_class=set()
        for cid,patch,mid in matched:
            unambiguous = mid>0 and mid in by_id and classes_by_mask.get(mid,set()) == {cid}
            if unambiguous and (cid,mid) not in used_by_class:
                vec=F.normalize(sam_weight*by_id[mid]["embedding"]+patch_weight*patch,dim=0)
                direct_anchors[mid]=cid
                used_by_class.add((cid,mid))
            elif unambiguous:
                # Repeated points on the same object do not create a second
                # prototype contribution or a second query target.
                continue
            else:
                vec=patch; patch_only.add(cid)
            values_by_class.setdefault(cid,[]).append(vec)
        for cid,values in values_by_class.items():
            if values: vectors[int(cid)]=F.normalize(torch.stack(values).mean(0),dim=0)
        return vectors,direct_anchors,patch_only

    radius=int(cfg.get("prompt",{}).get("point_match_radius",20))
    matched=[]; classes_by_mask={}
    for cid,class_points in points.items():
        for x,y in class_points:
            side=patch_side(cid)
            patch=point_patch_embedding(feature,x,y,image_size,side)
            mid=nearest_instance_id(inst,x,y,radius=radius)
            matched.append((int(cid),patch,int(mid)))
            if mid>0 and mid in by_id: classes_by_mask.setdefault(int(mid),set()).add(int(cid))
    vectors={}; direct_anchors={}; patch_only=set(); values_by_class={}
    for cid,patch,mid in matched:
        if mid>0 and mid in by_id and len(classes_by_mask.get(mid,set()))==1:
            vec=F.normalize(sam_weight*by_id[mid]["embedding"]+patch_weight*patch,dim=0)
            direct_anchors[mid]=cid
        else:
            vec=patch; patch_only.add(cid)
        values_by_class.setdefault(cid,[]).append(vec)
    for cid,values in values_by_class.items():
        if values: vectors[cid]=F.normalize(torch.stack(values).mean(0),dim=0)
    return vectors,direct_anchors,patch_only


class _UF:
    def __init__(self,ids): self.p={int(x):int(x) for x in ids}
    def find(self,x):
        x=int(x)
        while self.p[x]!=x: self.p[x]=self.p[self.p[x]]; x=self.p[x]
        return x
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.p[b]=a


def _leaves(tree):
    return [tree] if isinstance(tree,int) else _leaves(tree[0])+_leaves(tree[1])


def complete_graph_forest(nodes,threshold=0.8,max_area=0.25,image_area=1):
    by={int(n["mask_id"]):n for n in nodes}; ids=sorted(by); uf=_UF(ids)
    comp={i:{"area":float(by[i]["area"]),"sum":by[i]["embedding"]*float(by[i]["area"])} for i in ids}
    trees={i:i for i in ids}; edges=[]
    for aidx,a in enumerate(ids):
        for b in ids[aidx+1:]: edges.append((float(torch.dot(by[a]["embedding"],by[b]["embedding"])),a,b))
    for edge,a,b in sorted(edges,reverse=True):
        ra,rb=uf.find(a),uf.find(b)
        if ra==rb:continue
        ca,cb=comp[ra],comp[rb]
        if ca["area"]+cb["area"]>float(image_area)*float(max_area):continue
        cohesion=float(torch.dot(F.normalize(ca["sum"],dim=0),F.normalize(cb["sum"],dim=0)))
        if min(edge,cohesion)<float(threshold):continue
        ta,tb=trees[ra],trees[rb]; uf.union(ra,rb); root=uf.find(ra)
        comp[root]={"area":ca["area"]+cb["area"],"sum":ca["sum"]+cb["sum"]}; trees[root]=(ta,tb)
    return [trees[r] for r in sorted({uf.find(i) for i in ids})]


def refine_forest_with_points(forest,direct_anchors):
    groups=[]
    def cut(tree):
        leaves=_leaves(tree); classes={direct_anchors[m] for m in leaves if m in direct_anchors}
        if len(classes)<=1 or isinstance(tree,int): groups.append((leaves,classes)); return
        cut(tree[0]); cut(tree[1])
    for tree in forest:cut(tree)
    return groups


def _group_embedding(leaves,by):
    vec=[]; weights=[]
    for mid in leaves:
        if mid in by: vec.append(by[mid]["embedding"]); weights.append(float(by[mid]["area"])**0.5)
    if not vec:return None
    x=torch.stack(vec); w=torch.tensor(weights,device=x.device,dtype=x.dtype)
    return F.normalize((x*w[:,None]).sum(0)/w.sum().clamp_min(1e-6),dim=0)


def build_hybrid_pseudo(
    image, encoder, instance_map, points, class_ids, cfg, epoch=0,
    decoder_scores=None, cache_path=None, global_prototypes=None, feature=None,
):
    if feature is None:
        feature=rotation_consistent_feature(encoder,image,cache_path=cache_path)
    nodes=proposal_nodes(feature,instance_map); by={int(n["mask_id"]):n for n in nodes}
    local_prototypes,anchors,patch_only=adaptive_anchor_prototypes(feature,nodes,instance_map,points,image.size,cfg)
    pcfg=cfg.get("global_prototype",{})
    fcfg=pcfg.get("fusion",{})
    prototypes=resolve_class_prototypes(
        local_prototypes, global_prototypes,
        enabled=bool(pcfg.get("enabled",False)),
        fusion_enabled=bool(fcfg.get("enabled",False)),
        local_weight=float(fcfg.get("local_weight",0.5)),
        global_weight=float(fcfg.get("global_weight",0.5)),
    )
    gcfg=cfg.get("complete_graph",{})
    if bool(gcfg.get("enabled",True)):
        forest=complete_graph_forest(nodes,gcfg.get("threshold",0.8),gcfg.get("max_area_ratio",0.25),np.asarray(instance_map).size)
        groups=refine_forest_with_points(forest,anchors)
    else:
        groups=[([int(node["mask_id"])],({anchors[int(node["mask_id"])]} if int(node["mask_id"]) in anchors else set())) for node in nodes]
    records=[]; assigned=set(); accept=float(gcfg.get("class_threshold",0.55)); margin=float(gcfg.get("class_margin",0.05))
    for gid,(leaves,classes) in enumerate(groups,1):
        emb=_group_embedding(leaves,by)
        if emb is None:continue
        if len(classes)==1:
            cid=next(iter(classes)); conf=1.0; source="point_group"
        else:
            scores=[]
            for cid,proto in prototypes.items():
                score=float(torch.dot(emb,proto))
                if decoder_scores:
                    decoder_values = [decoder_scores.get(m,{}).get(cid,0.0) for m in leaves]
                    decoder_value = float(np.nan_to_num(np.mean(decoder_values), nan=0.0, posinf=0.0, neginf=0.0))
                    score += float(gcfg.get("decoder_weight",0.25)) * decoder_value
                if np.isfinite(score): scores.append((score,cid))
            scores.sort(reverse=True)
            if not scores:continue
            second=scores[1][0] if len(scores)>1 else -1.0
            if scores[0][0]<accept or scores[0][0]-second<margin:continue
            conf=float(np.clip(scores[0][0],0,1)); cid=int(scores[0][1]); source="dino_group"
        for mid in leaves:
            if mid in assigned:continue
            records.append(PseudoInstance(int(mid),int(cid),conf,source,int(mid) in anchors,epoch,epoch,group_id=int(gid))); assigned.add(int(mid))
    # Optional dataset-specific safeguard: tiny human proposals are not
    # reliable enough to be propagated from DINO/group assignments into the
    # persistent instance bank.  Keep direct SAM point anchors, but remove
    # only the selected classes' unanchored propagated instances.
    excluded=set(int(x) for x in cfg.get("point_supervision", {}).get("exclude_propagated_classes", []))
    if excluded:
        records=[r for r in records if not (int(r.class_id) in excluded and r.source == "dino_group")]
    available=[c for c in class_ids if int(c) in prototypes]
    if available:
        p=torch.stack([prototypes[int(c)] for c in available]); logits=torch.einsum("bchw,kc->bkhw",feature,p)
        logits=F.interpolate(logits,size=(image.height,image.width),mode="bilinear",align_corners=False); idx=logits.argmax(1)[0].cpu().numpy()
        semantic=np.zeros((image.height,image.width),dtype=np.uint16)
        for i,cid in enumerate(available):semantic[idx==i]=int(cid)
        confidence=logits.softmax(1).max(1).values[0].cpu().numpy()
    else:
        semantic=np.full((image.height,image.width),255,dtype=np.uint16); confidence=np.zeros_like(semantic,dtype=np.float32)
    point_fallbacks=[]
    pscfg=cfg.get("point_supervision", {})
    if bool(pscfg.get("enabled", False)):
        # A point that did not become a SAM anchor remains valid supervision.
        # It gets a tiny instance target and a 3x3 semantic target.
        inst=np.asarray(instance_map)
        semantic_radius=max(0, (int(pscfg.get("semantic_size", 3)) - 1) // 2)
        for cid,class_points in points.items():
            for x,y in class_points:
                xi=int(np.clip(round(float(x)),0,inst.shape[1]-1)); yi=int(np.clip(round(float(y)),0,inst.shape[0]-1))
                mid=int(inst[yi,xi])
                if mid <= 0 or mid not in anchors:
                    point_fallbacks.append({"class_id":int(cid),"x":float(x),"y":float(y),"confidence":1.0})
                x0,x1=max(0,xi-semantic_radius),min(image.width,xi+semantic_radius+1)
                y0,y1=max(0,yi-semantic_radius),min(image.height,yi+semantic_radius+1)
                semantic[y0:y1,x0:x1]=int(cid); confidence[y0:y1,x0:x1]=1.0
    return {"feature":feature,"nodes":nodes,"records":records,"semantic":semantic,"semantic_confidence":confidence,"prototypes":prototypes,"local_prototypes":local_prototypes,"anchors":anchors,"patch_only_classes":patch_only,"groups":groups,"point_fallbacks":point_fallbacks}
