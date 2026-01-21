import pathlib
import os


def load_dotenv():
    from dotenv import load_dotenv
    env_state = os.getenv("ENV_STATE",'dev')
    load_dotenv()
    # load_dotenv(f'.env.{env_state}',override=True)

load_dotenv()

PYTHON3 = os.getenv("PYTHON3")