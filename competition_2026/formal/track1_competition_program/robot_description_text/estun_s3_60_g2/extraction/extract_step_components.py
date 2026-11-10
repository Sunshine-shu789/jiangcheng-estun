import json
import math
import os
import sys
import time
from collections import defaultdict

from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.Bnd import Bnd_Box
from OCP.GeomAbs import GeomAbs_SurfaceType
from OCP.GProp import GProp_GProps
from OCP.StlAPI import StlAPI_Writer
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label, TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ShapeTool

step_path = sys.argv[1]
out_dir = sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

started = time.time()
doc = TDocStd_Document(TCollection_ExtendedString("doc"))
reader = STEPCAFControl_Reader()
reader.SetNameMode(True)
reader.SetColorMode(True)
reader.SetMatMode(True)
assert reader.ReadFile(step_path)
assert reader.Transfer(doc)
print(f"import_s={time.time()-started:.3f}", flush=True)
tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())


def label_name(label):
    attr = TDataStd_Name()
    return attr.Get().ToExtString() if label.FindAttribute(TDataStd_Name.GetID_s(), attr) else ""


def bbox(shape):
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, box, False, False)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return [xmin, xmax, ymin, ymax, zmin, zmax]


def count(shape, typ):
    exp = TopExp_Explorer(shape, typ)
    result = 0
    while exp.More():
        result += 1
        exp.Next()
    return result


def canonical_direction(x, y, z):
    values = [x, y, z]
    largest = max(range(3), key=lambda i: abs(values[i]))
    if values[largest] < 0:
        values = [-v for v in values]
    return tuple(values)


def cylinders(shape):
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    grouped = defaultdict(lambda: {"area": 0.0, "faces": 0, "radii": set()})
    all_faces = []
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        surf = BRepAdaptor_Surface(face, True)
        if surf.GetType() == GeomAbs_SurfaceType.GeomAbs_Cylinder:
            cyl = surf.Cylinder()
            axis = cyl.Axis()
            loc = axis.Location()
            direction = axis.Direction()
            dx, dy, dz = direction.X(), direction.Y(), direction.Z()
            d = canonical_direction(dx, dy, dz)
            if d != (dx, dy, dz):
                direction.Reverse()
                dx, dy, dz = -dx, -dy, -dz
            props = GProp_GProps()
            BRepGProp.SurfaceProperties_s(face, props)
            area = abs(props.Mass())
            # Perpendicular offset of the axis from the assembly origin.
            px, py, pz = loc.X(), loc.Y(), loc.Z()
            projection = px * dx + py * dy + pz * dz
            ox, oy, oz = px - projection * dx, py - projection * dy, pz - projection * dz
            key = (
                round(dx, 5), round(dy, 5), round(dz, 5),
                round(ox, 3), round(oy, 3), round(oz, 3),
            )
            group = grouped[key]
            group["area"] += area
            group["faces"] += 1
            group["radii"].add(round(cyl.Radius(), 3))
            all_faces.append({
                "axis_point_mm": [round(px, 4), round(py, 4), round(pz, 4)],
                "axis_dir": [round(dx, 7), round(dy, 7), round(dz, 7)],
                "axis_offset_mm": [round(ox, 4), round(oy, 4), round(oz, 4)],
                "radius_mm": round(cyl.Radius(), 4),
                "area_mm2": round(area, 3),
            })
        exp.Next()
    groups = []
    for key, group in grouped.items():
        groups.append({
            "axis_dir": list(key[:3]),
            "axis_offset_mm": list(key[3:]),
            "faces": group["faces"],
            "area_mm2": round(group["area"], 3),
            "radii_mm": sorted(group["radii"]),
        })
    groups.sort(key=lambda item: (-item["area_mm2"], -max(item["radii_mm"])))
    return groups


roots = TDF_LabelSequence()
tool.GetFreeShapes(roots)
report = {"step": step_path, "root": label_name(roots.Value(1)), "components": []}
for root_index in range(1, roots.Length() + 1):
    root = roots.Value(root_index)
    components = TDF_LabelSequence()
    XCAFDoc_ShapeTool.GetComponents_s(root, components, False)
    for index in range(1, components.Length() + 1):
        occurrence = components.Value(index)
        referred = TDF_Label()
        XCAFDoc_ShapeTool.GetReferredShape_s(occurrence, referred)
        located = XCAFDoc_ShapeTool.GetShape_s(occurrence)
        local = XCAFDoc_ShapeTool.GetShape_s(referred)
        location = XCAFDoc_ShapeTool.GetLocation_s(occurrence).Transformation()
        matrix = [[round(location.Value(i, j), 9) for j in range(1, 5)] for i in range(1, 4)]
        entry = {
            "index": index,
            "occurrence": label_name(occurrence),
            "part": label_name(referred),
            "solids": count(located, TopAbs_SOLID),
            "faces": count(located, TopAbs_FACE),
            "located_bbox_mm": [round(v, 6) for v in bbox(located)],
            "local_bbox_mm": [round(v, 6) for v in bbox(local)],
            "matrix": matrix,
            "cylinders": cylinders(located),
        }
        report["components"].append(entry)
        print(json.dumps({k: entry[k] for k in ("index", "occurrence", "part", "solids", "faces", "located_bbox_mm")}, ensure_ascii=False), flush=True)
        stl_path = os.path.join(out_dir, f"NAUO{index}.stl")
        mesh_started = time.time()
        BRepMesh_IncrementalMesh(located, 0.10, False, 0.20, True)
        writer = StlAPI_Writer()
        writer.ASCIIMode = False
        writer.Write(located, stl_path)
        entry["stl"] = stl_path
        entry["stl_bytes"] = os.path.getsize(stl_path)
        print(f"  STL {stl_path} bytes={entry['stl_bytes']} s={time.time()-mesh_started:.3f}", flush=True)

with open(os.path.join(out_dir, "step_report.json"), "w", encoding="utf-8") as handle:
    json.dump(report, handle, ensure_ascii=False, indent=2)
print(f"TOTAL s={time.time()-started:.3f}", flush=True)
