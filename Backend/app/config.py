from pydantic import BaseModel


class Settings(BaseModel):
    # Threshold statis — TIDAK BOLEH berubah saat runtime (aturan panitia R7.4)
    T_ANOMALY: float = 0.75      # ambang skor anomali dari EfficientAD
    T_AREA: float = 2.0          # ambang luas cacat dalam persen
    T_CONF: float = 0.60         # ambang keyakinan minimum model

    # Validasi upload gambar
    MAX_FILE_SIZE_MB: int = 10
    ALLOWED_CONTENT_TYPES: tuple[str, ...] = ("image/jpeg", "image/png")


settings = Settings()