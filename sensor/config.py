from dataclasses import dataclass
import os
import pymongo



@dataclass


class EnvironmentVariable:
    mongo_db_url = os.getenv("MONGODB_URL_KEY")



env_var = EnvironmentVariable()

mongo_client = pymongo.MongoClient(env_var.mongo_db_url)



