from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    OPENAI_API_KEY: str
    DLPFC_MODEL: str = "gpt-4"
    VMPFC_MODEL: str = "gpt-4"
    OFC_MODEL: str = "gpt-4"
    ACC_MODEL: str = "gpt-4"
    MPFC_MODEL: str = "gpt-4"


settings = Settings()  # type: ignore
