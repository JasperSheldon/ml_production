from typing import Any, List,Optional

from pydantic import BaseModel
from classification_model.processing.validation import TitanicDataInputSchema


class PredictionResults(BaseModel):
    errors:Optional[Any]
    version:str
    predictions:Optional[int]
    probability:Optional[Any]
    


class MultipleTitanicInputs(BaseModel):
    inputs:List[TitanicDataInputSchema]

    class Config:
        schema_extra={
            "example":{
                "inputs":[
                    {
                        "pclass": 1,
                        "name": "Allison, Miss. Helen Loraine",
                        "sex": "female",
                        "age": 2,
                        "sibsp": 1,
                        "parch": 2,
                        "ticket": 113781,
                        "fare": 151.55,
                        "cabin": "C22",
                        "embarked": "S",
                        "boat": None,
                        "body": None

                    }
                ]
            }
        }

