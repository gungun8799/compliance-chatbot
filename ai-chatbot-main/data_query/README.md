# Chat Log Synchronization Script

This script syncs chat logs from a PostgreSQL database, processes them, and uploads the extracted data as CSV files to a SharePoint folder. The synchronization is done within a specified date range.

## Prerequisites

- Python 3.x
- PostgreSQL database
- SharePoint access
- SSH access to the server

## Installation

1. Clone the repository or download the script files.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the `.env` file is updated with the correct configurations.

## Environment Variables

Create a `.env` file in the root directory and populate it with the following variables:

```
TIMEZONE=Asia/Bangkok
SYNC_START_DATE=2024-01-01  # (optional, format: YYYY-MM-DD)
SYNC_END_DATE=2024-01-31    # (optional, format: YYYY-MM-DD)

POSTGRES_DB=mydatabase
POSTGRES_USERNAME=myuser
POSTGRES_PASSWORD=mypassword
POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5432
```

## Steps to Run the Script

1. **Tunnel to the server** (if accessing a remote database):
   ```bash
   ssh -L 5432:127.0.0.1:5432 jittagorn.pitakmetagoon@10.2.192.15
   ```
2. **Ensure the `.env` file is correctly updated** with database and SharePoint credentials.
3. **Run the script:**
   ```bash
   python3 main.py
   ```

## Script Workflow

1. Reads the start and end dates from the `.env` file or defaults to the current date.
2. Fetches employee IDs from the PostgreSQL database for the given date range.
3. Retrieves messages for each employee and processes them.
4. Saves the data as CSV files.
5. Uploads the CSV files to SharePoint.
6. Deletes temporary CSV files after upload.

## Logging

The script logs important information and errors in the console using Python's `logging` module.

## File Structure

```
.
├── main.py                  # Main script file
├── query/
│   ├── get_employee_ids.sql # SQL query to get employee IDs
│   ├── get_messages.sql     # SQL query to get chat messages
├── .env                     # Environment variables (not included in repo)
└── requirements.txt          # Python dependencies
```

## Troubleshooting

- **Connection issues to PostgreSQL?**
  - Ensure the SSH tunnel is set up correctly.
  - Check `.env` for correct database credentials.
- **SharePoint upload issues?**
  - Verify access token retrieval is working.
  - Ensure folder permissions are correct.

## License

This project is for internal use only. All rights reserved.

---

**Author:** Chirawat Chitpakdee

