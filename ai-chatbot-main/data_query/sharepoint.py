import logging
from dotenv import load_dotenv
import os
import requests
import urllib3

if os.path.exists(".env"):
    load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]  %(message)s"
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

domain_name = os.getenv("SHAREPOINT_DOMAIN_NAME")
tenant_id = os.getenv("SHAREPOINT_TENANT_ID")

token_url = f"https://accounts.accesscontrol.windows.net/{tenant_id}/tokens/OAuth/2/"
client_id = f"{os.getenv('SHAREPOINT_OAUTH_CLIENT_ID')}@{tenant_id}"
client_secret = os.getenv("SHAREPOINT_OAUTH_CLIENT_SECRET")
resource = f"00000003-0000-0ff1-ce00-000000000000/{domain_name}@{tenant_id}"

base_url = f"https://{domain_name}"
site_name = os.getenv("SHAREPOINT_SITE_NAME")
parent_folder = os.getenv("SHAREPOINT_PARENT_FOLDER")


def get_access_token():
    """
    Function to get an OAuth access token for access the Microsoft Sharepoint
    """

    try:
        response = requests.post(
            url=token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
                "resource": resource,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if response.status_code == 200:
            return response.json().get("access_token")
        else:
            logger.error(
                f"Get access token error: {response.status_code} - {response.text}"
            )
    except Exception as e:
        logger.error(f"Get access token error: {e}")


def folder_exists(access_token, folder):
    """
    Function to check the target folder is exists on Microsoft Sharepoint or not
    """

    try:
        url = f"{base_url}{site_name}/_api/Web/GetFolderByServerRelativeUrl('{site_name}{parent_folder}/{folder}')"
        logger.info(f'Checking folder "{folder}" at {url}')

        response = requests.get(
            url=url,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json;odata=verbose",
            },
        )
        is_exists = response.status_code == 200
        if is_exists:
            logger.info(f'Folder "{folder}" already exists')
        else:
            logger.info(f'Folder "{folder}" not exists')
        return is_exists
    except Exception as e:
        logger.error(f'Check folder "{folder}" exists error: {e}')


def create_folder(access_token, folder):
    """
    Function to create a folder on Microsoft Sharepoint
    """

    try:
        response = requests.post(
            url=f"{base_url}{site_name}/_api/web/folders",
            data=str(
                {
                    "__metadata": {"type": "SP.Folder"},
                    "ServerRelativeUrl": f"{site_name}{parent_folder}/{folder}",
                }
            ),
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json;odata=verbose",
            },
        )
        if response.status_code == 201:
            logger.info(f'Create folder "{folder}" successfully')
        else:
            logger.error(
                f'Create folder "{folder}" error: {response.status_code} - {response.text}'
            )
    except Exception as e:
        logger.error(f'Create folder "{folder}" error: {e}')


def upload_file(access_token, folder, file_name):
    """
    Function to upload file to the target folder on Microsoft Sharepoint
    """

    try:
        logger.info(f'Uploading file "{file_name}" to folder "{folder}"...')
        with open(file_name, "rb") as file:
            response = requests.post(
                url=f"{base_url}{site_name}/_api/Web/GetFolderByServerRelativeUrl('{site_name}{parent_folder}/{folder}')/Files/add(url='{file_name}',overwrite=true)",
                data=file,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/json;odata=verbose",
                },
            )

        if response.status_code == 200:
            logger.info(f'Upload file "{file_name}" successfully')
        else:
            logger.error(
                f'Upload file "{file_name}" error: {response.status_code} - {response.text}'
            )
    except Exception as e:
        logger.error(f'Upload file "{file_name}" error: {e}')
