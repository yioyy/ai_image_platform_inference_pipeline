import pathlib
import os


def load_dotenv():
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return

    env_state = os.getenv("ENV_STATE", "dev")
    del env_state  # 預留：未來可依環境載入 .env.<env_state>
    load_dotenv()
    # load_dotenv(f'.env.{env_state}',override=True)

load_dotenv()

PYTHON3 = os.getenv("PYTHON3")