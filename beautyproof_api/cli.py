import argparse,json
from .unified import UnifiedBeautyProofAPI

def main():
    p=argparse.ArgumentParser(description="BeautyProof unified analysis")
    p.add_argument("image");p.add_argument("--output")
    a=p.parse_args();result=UnifiedBeautyProofAPI().analyze(a.image);text=json.dumps(result,ensure_ascii=False,indent=2)
    if a.output: open(a.output,"w",encoding="utf-8").write(text)
    print(text)
if __name__=="__main__":main()
