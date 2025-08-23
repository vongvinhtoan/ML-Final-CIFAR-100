from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import torch
from configs import settings
from pydantic import BaseModel
from torch import nn


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
    
    def _get_time_stamp(self) -> str:
        return datetime.now().strftime("%y-%m-%d-%H-%M-%S")

    def upload_weight(self, model: nn.Module) -> str:
        # build filename and save locally
        filename = f"{self._get_time_stamp()}.pth"
        filepath = settings.WEIGHT_PATH / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), filepath)

        # upload to Google Drive
        file_metadata = {
            "name": filename,
            "parents": [str(self.folder_id)]
        }
        media = MediaFileUpload(str(filepath), mimetype="application/octet-stream")
        uploaded = service.files().create(body=file_metadata, media_body=media, fields="id").execute()

        file_id = uploaded.get("id")
        return f"https://drive.google.com/file/d/{file_id}"


ggdrive = GGDrive()