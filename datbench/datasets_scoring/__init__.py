"""Dataset-specific scoring modules.
"""
from . import (
    pixmo_points_eval, docvqa, ocrvqa, countbench, refcoco, textvqa, vqav2, tallyqa,
    chartqa, infovqa, ai2d, mmbench, realworldqa, mmmu, mmmupro, mathvista, mathverse,
    mathvision, logicvista, charxiv, cc_ocr_multi_scene_ocr, mme_realworld, chartqapro,
    cc_ocr_doc_parsing, cc_ocr_kie, ocrbench_v2
)

# Dataset registry: maps dataset name → scoring module
DATASET_SCORING_MODULES = {
    # Grounding
    'pixmo_points_eval': pixmo_points_eval,
    'refcoco_testA': refcoco,
    'refcoco_testB': refcoco,
    'refcoco_plus_testA': refcoco,
    'refcoco_plus_testB': refcoco,
    'refcoco_m_val': refcoco,
    'refcocog_test': refcoco,

    # Document
    'docvqa': docvqa,
    'ocr-vqa': ocrvqa,
    'infovqa': infovqa,
    'cc-ocr-doc_parsing': cc_ocr_doc_parsing,
    'cc-ocr-kie': cc_ocr_kie,
    'ocrbench-v2': ocrbench_v2,

    # Counting
    'countbench': countbench,
    'tallyqa': tallyqa,

    # Scene (TextVQA)
    'text-vqa': textvqa,
    'cc-ocr-multi_scene_ocr': cc_ocr_multi_scene_ocr,

    # General (VQAv2)
    'vqa-v2': vqav2,
    'ai2d': ai2d,
    'mmbench': mmbench,
    'mmmu': mmmu,
    'mmmupro': mmmupro,

    # Spatial
    'realworldqa': realworldqa,
    'mme-realworld-ocr': mme_realworld,
    'mme-realworld-ad': mme_realworld,
    'mme-realworld-mo': mme_realworld,
    'mme-realworld-dt': mme_realworld,

    # Chart
    'chartqa': chartqa,
    'chartqapro': chartqapro,
    'charxiv_descriptive': charxiv,
    'charxiv_reasoning': charxiv,

    # Math
    'mathvista': mathvista,
    'mathverse_reasoning': mathverse,
    'mathverse_wo': mathverse,
    'mathvision': mathvision,
    'logicvista': logicvista,
}

__all__ = ['DATASET_SCORING_MODULES']
