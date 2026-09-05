from beautyproof_api import UnifiedBeautyProofAPI
from beautyproof_api.production import type_predict

def test_all_three_real_checkpoints_load_and_public_schema_filters_experimental_outputs():
    result=UnifiedBeautyProofAPI().analyze("demo/vendor/ai-image-detector/test_images/ai_retouched.png")
    assert isinstance(result["retouched"],bool)
    assert 0 <= result["retouch_probability"] <= 1
    assert all(x["name"] in {"skin_enhancement","face_slimming","facial_contouring"} for x in result["retouch_types"])
    assert all(x["name"] != "jawline" for x in result["modified_regions"])


def test_real_three_type_checkpoint_loads():
    result=type_predict("demo/vendor/ai-image-detector/test_images/ai_retouched.png")
    assert set(result["probabilities"]) == {"skin_enhancement","face_slimming","facial_contouring"}
    assert all(0 <= value <= 1 for value in result["probabilities"].values())
