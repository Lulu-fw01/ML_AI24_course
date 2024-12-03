import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager


class Item(BaseModel):
    year: int
    km_driven: int
    mileage_kmpl: float
    engine_cc: int
    max_power_bhp: float
    seats: int

class Items(BaseModel):
    objects: List[Item]

ml_models = {}

def medinc_regressor(x: dict) -> dict:
    with open("model.pkl", 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    x_df = pd.DataFrame(x, index=[0])
    res = loaded_model.predict(x_df)[0]
    return {"prediction": res}


@asynccontextmanager
async def ml_lifespan_manager(app: FastAPI):
    ml_models["my_model"] = medinc_regressor
    yield
    ml_models.clear()

app = FastAPI(lifespan=ml_lifespan_manager)


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    return ml_models["gc"](item.model_dump())


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    return ...



# from fastapi import FastAPI
# from fastapi.responses import StreamingResponse
# import pandas as pd
# import io


# app = FastAPI()

# @app.get("/csv")
# def read_csv():
#     df = pd.DataFrame({'Name': ['John', 'Anna', 'Peter'],
#                        'Age': [28, 24, 35]})
    
#     stream = io.StringIO()
#     df.to_csv(stream, index=False)
#     response = StreamingResponse(iter([stream.getvalue()]),
#                                  media_type="text/csv"
#                                  )
    
#     response.headers["Content-Disposition"] = "attachment; filename=result.csv"
    
#     return response