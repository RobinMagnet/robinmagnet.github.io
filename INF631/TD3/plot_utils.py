# plot_utils.py
import trimesh
import numpy as np
from scipy.spatial import cKDTree as KDTree


def trimesh_from_vfc(v,f,c):
    """
    Create trimesh from vertices, faces and colors

    Args:
        v (np.ndarray): vertices
        f (np.ndarray): faces
        c (np.ndarray): colors

    Returns:
        m (trimesh.Trimesh): trimesh object
    """
    m = trimesh.Trimesh(vertices=v, faces=f, vertex_colors=c, process=False)
    return m

def check_is_trimesh(t_obj):
    if isinstance(t_obj, trimesh.Trimesh) or isinstance(t_obj, trimesh.points.PointCloud):
        return True

def v_from_m(m):
    if isinstance(m, str):
        return np.array(trimesh.load(m, process=False).vertices)
    elif isinstance(m, np.ndarray):
        # assume it's vert already
        return m
    elif check_is_trimesh(m):
        return np.array(m.vertices)
    elif isinstance(m, trimesh.Scene):
        return np.array(m.dump().sum().vertices)
    else:
        raise ValueError("Unknown format")
    
def f_from_m(m):
    m = trimesh.load(m, process=False)
    if isinstance(m, trimesh.Scene):
        return np.array(m.dump().sum().faces)
    return np.array(m.faces)
    
def scale_to_unit_sphere(points):
    """
    Scale vertices to unit sphere

    Args:
        points (np.ndarray): vertices
        
    Returns:
        points (np.ndarray): vertices scaled to unit sphere
    """
    midpoints = (np.max(points, axis=0) + np.min(points, axis=0)) / 2
#     midpoints = np.mean(points, axis=0)
    points = points - midpoints
    scale = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    points = points / scale
    return points

def plot_segmentation_mesh(verts, faces, segments):
    """
    Plot segmentation mesh

    Args:
        verts (np.ndarray): vertices
        faces (np.ndarray): faces
        segments (np.ndarray): segments as integers

    Returns:
        mesh (trimesh.Trimesh): trimesh object with colors corresponding to segments
    """
    assert verts.shape[0]==segments.shape[0]
    colors = trimesh.visual.interpolate(np.arange(np.unique(segments).shape[0]), color_map='jet')
    all_vert, all_color = [], []
    for (seg_id, color) in zip(np.unique(segments), colors):
        seg_vert = verts[segments==seg_id]
        all_vert.append(seg_vert)
        all_color.append(np.repeat(color[np.newaxis, :], seg_vert.shape[0], axis=0))
    tree = KDTree(np.vstack(all_vert))
    nn_ind = tree.query(verts)[1]
    mesh = trimesh_from_vfc(verts, np.array(faces), np.vstack(all_color)[nn_ind])    
    return mesh

def plot_multi_meshes(meshes, cmap='vert_colors',
                      spacing=[1.5, 0., 0.],
                      scale=True):
    
    """
    Juxtapose multiple meshes
    
    Returns:
        meshes: list of meshes to juxtapose
        cmap: color map corresponding to segmentations
        spacing: spacing between meshes
        
    Returns:
        scene: trimesh scene object
    """
    s1 = trimesh.Scene()
    assert isinstance(meshes, list), 'Only lists'
    if cmap.lower() != 'vert_colors':
        assert isinstance(cmap, list), 'Only lists'

    n_m = len(meshes)
    spacing = [np.array(spacing)*i for i in range(1, n_m+1)]
    
    for ind, m in enumerate(meshes):
        if isinstance(m, str):
            m = trimesh.load(m, process=False)
        assert isinstance(m, trimesh.Trimesh), "should be a mesh or str"
        if cmap is not None and cmap.lower() == 'vert_colors':
            color = m.visual.vertex_colors
        else:
            color = cmap[ind]
        verts = scale_to_unit_sphere(v_from_m(m)) if scale else v_from_m(m)
        m_new = trimesh_from_vfc(verts+spacing[ind], f_from_m(m), color)
        s1.add_geometry(m_new)

    return s1