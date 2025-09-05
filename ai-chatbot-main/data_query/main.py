import logging
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import pytz
import sharepoint
import csv
import psycopg2
from psycopg2.extras import DictCursor

if os.path.exists(".env"):
    load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]  %(message)s"
)

TIMEZONE = os.getenv("TIMEZONE")
START_DATE = os.getenv("SYNC_START_DATE")
END_DATE = os.getenv("SYNC_END_DATE")

db_params = {
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USERNAME"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host": os.getenv("POSTGRES_HOST"),
    "port": os.getenv("POSTGRES_PORT"),
}

timezone = pytz.timezone(TIMEZONE)


def to_start_of_day(date):
    return date.replace(hour=0, minute=0, second=0, microsecond=0)


def to_end_of_day(date):
    return date.replace(hour=23, minute=59, second=59, microsecond=0)


def get_start_date():
    if START_DATE is None:
        return to_start_of_day(datetime.now(timezone))
    else:
        return to_start_of_day(
            timezone.localize(datetime.strptime(START_DATE, "%Y-%m-%d"))
        )


def get_end_date():
    if END_DATE is None:
        return to_end_of_day(datetime.now(timezone))
    else:
        return to_end_of_day(timezone.localize(datetime.strptime(END_DATE, "%Y-%m-%d")))


def upload_file_to_sharepoint(access_token, folder, file_name):
    if not sharepoint.folder_exists(access_token=access_token, folder=folder):
        sharepoint.create_folder(access_token=access_token, folder=folder)

    sharepoint.upload_file(
        access_token=access_token, folder=folder, file_name=file_name
    )


def read_file_content(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def write_data_to_csv(column_names, rows, output_file):
    with open(output_file, mode="w", newline="", encoding="utf-8-sig") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(column_names)
        csv_writer.writerows(rows)


def get_employee_ids(start_date, end_date):
    try:
        connection = psycopg2.connect(**db_params)
        cursor = connection.cursor(cursor_factory=DictCursor)

        query = read_file_content("query/get_employee_ids.sql")
        cursor.execute(
            query,
            {
                "start_date": start_date,
                "end_date": end_date,
            },
        )

        results = cursor.fetchall()
        return [row["employee_id"] for row in results]
    except psycopg2.Error as e:
        logger.error(f"get_employee_ids error: {e}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def get_messages(start_date, end_date, employee_id):
    try:
        connection = psycopg2.connect(**db_params)
        cursor = connection.cursor(cursor_factory=DictCursor)

        query = read_file_content("query/get_messages.sql")
        cursor.execute(
            query,
            {
                "start_date": start_date,
                "end_date": end_date,
                "employee_id": employee_id,
                "timezone": TIMEZONE,
            },
        )

        return cursor.fetchall()
    except psycopg2.Error as e:
        logger.error(f"get_messages error: {e}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def modify_messages(messages):
    for index in range(1, len(messages)):
        message = messages[index]
        if message["role"] == "RUN":
            if messages[index + 1]["role"] == "ASSISTANT":
                messages[index + 1]["feedback_value"] = message["feedback_value"]
                messages[index + 1]["feedback_comment"] = message["feedback_comment"]
    return [message for message in messages if message["role"] != "RUN"]


def sync():
    try:
        start_date = get_start_date()
        end_date = get_end_date()

        logger.info("Sync chat logs with conditions:")
        logger.info(f"- Start date: {start_date}")
        logger.info(f"- End date: {end_date}")

        access_token = sharepoint.get_access_token()

        current_date = start_date
        while current_date <= end_date:
            start = to_start_of_day(current_date)
            end = to_end_of_day(current_date)

            employee_ids = get_employee_ids(start, end)

            if len(employee_ids) == 0:
                logger.info(f"No data from {start} to {end}")
            else:
                for employee_id in employee_ids:
                    messages = get_messages(start, end, employee_id)
                    messages = modify_messages(messages)

                    column_names = [
                        "Employee ID",
                        "Email",
                        "Sesion ID",
                        "Sesion Name",
                        "Role",
                        "Time",
                        "Message",
                        "Feedback Value",
                        "Feedback Comment",
                    ]

                    csv_file = f"{employee_id}.csv"
                    try:
                        write_data_to_csv(column_names, messages, csv_file)
                        upload_file_to_sharepoint(
                            access_token, start.strftime("%Y-%m-%d"), csv_file
                        )
                    finally:
                        if os.path.exists(csv_file):
                            os.remove(csv_file)

            current_date += timedelta(days=1)
        logger.info("Sync completed.")
    except Exception as e:
        logger.error(f"Sync chat log error: {e}")


def main():
    sync()


if __name__ == "__main__":
    main()