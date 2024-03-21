from fastapi import FastAPI, Query
from gensim.models import KeyedVectors, Word2Vec
from typing import List
import boto3

app = FastAPI()

s3 = boto3.client('s3')

# 버킷 이름과 오브젝트 키(파일 이름) 설정
bucket_name = 'elasticbeanstalk-us-east-1-905418103132'
object_key = 'recipe_word2vec_model.model'

# 파일을 임시로 저장할 로컬 경로
local_file_name = '/tmp/recipe_word2vec_model.model'

# S3 버킷에서 파일 다운로드
s3.download_file(bucket_name, object_key, local_file_name)

# 다운로드한 파일을 사용하여 모델 로드
from gensim.models import Word2Vec
model = Word2Vec.load(local_file_name).wv


# model = Word2Vec.load("/Users/choejong-gyu/Downloads/recipe_word2vec_model.model").wv
@app.get("/recommendations/")
async def get_recommendations(ingredients: List[str] = Query(None)):
   
    similar_recipes = model.most_similar(ingredients[0], topn=10)
    return {"recommendations": similar_recipes}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
