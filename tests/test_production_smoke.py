from beautyproof_api import UnifiedBeautyProofAPI

def test_all_three_real_checkpoints_load_and_public_schema_filters_experimental_outputs():
    result=UnifiedBeautyProofAPI().analyze("demo/vendor/ai-image-detector/test_images/ai_retouched.png")
    assert isinstance(result["retouched"],bool)
    assert 0 <= result["retouch_probability"] <= 1
    assert all(x["name"] in {"smoothing","whitening"} for x in result["retouch_types"])
    assert all(x["name"] != "jawline" for x in result["modified_regions"])
