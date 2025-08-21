from google.oauth2 import service_account
from googleapiclient.discovery import build
from .configs import settings
from pydantic import BaseModel

SCOPES = ["https://www.googleapis.com/auth/drive"]

creds = service_account.Credentials.from_service_account_file(settings.GGDRIVE_TOKEN_PATH, scopes=SCOPES)
service = build('drive', 'v3', credentials=creds)


class GGDriveFile(BaseModel):
    name: str | None = None
    id: str | None = None

    @property
    def url(self) -> str | None:
        if self.id:
            return f"https://drive.google.com/file/d/{self.id}"
        return None


class GGDrive:
    def __init__(self):
        self.folder_id = settings.GGDRIVE_FOLDER_ID

    def ls(self) -> list[GGDriveFile]:
        query = f"'{self.folder_id}' in parents"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        items = results.get('files', [])

        return [GGDriveFile(**item) for item in items]

    def upload_weight(self, ) -> str:
        return ""

ggdrive = GGDrive()