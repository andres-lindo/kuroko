"""Custom logging handler that ships log records to Azure Blob Storage.

Uses append blobs so that multiple processes can write to the same log file
without overwriting each other. Rotates to a new dated blob at midnight.
"""

import logging
from datetime import date
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError


class AzureBlobHandler(logging.Handler):
    """Logging handler that writes records to an Azure Blob Storage append blob.

    Each log entry is appended to a daily blob named
    ``{blob_name}_{YYYY-MM-DD}.log``. A new blob is created automatically
    when the calendar date changes, so log files never span more than one day.

    Attributes:
        connection_string: Azure Storage connection string used to
            authenticate all blob operations.
        container_name: Name of the blob container (lowercased on init).
        base_blob_name: Prefix used when constructing the dated blob name.
        blob_service_client: SDK client for the storage account.
        current_date: The calendar date of the currently active blob.
        blob_client: SDK client for the currently active append blob.
    """

    def __init__(self, connection_string, blob_name, container_name="logs"):
        """Initialise the handler and ensure the container and blob exist.

        Args:
            connection_string: Azure Storage connection string.
            blob_name: Base name for the log blob (date suffix is appended
                automatically, e.g. ``myapp`` becomes
                ``myapp_2024-01-15.log``).
            container_name: Target container name. Defaults to ``'logs'``.
                The value is lowercased because Azure requires lowercase
                container names.
        """
        super().__init__()
        self.connection_string = connection_string
        self.container_name = container_name.lower()
        self.base_blob_name = blob_name
        self.blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )

        # Create the container on first use; ignore the error if it already exists.
        try:
            self.blob_service_client.create_container(self.container_name)
        except ResourceExistsError:
            pass

        # Initialise state for today's blob so the first emit() call is cheap.
        self.current_date = date.today()
        self.blob_client = self._create_blob_client()

    def _create_blob_client(self):
        """Build a blob client for today's dated log file.

        The blob is created as an append blob if it does not already exist.
        Append blobs are the only blob type that supports concurrent writers
        without requiring a lock, which is why they are used here.

        Returns:
            BlobClient pointed at the log blob for the current date.
        """
        blob_name = f"{self.base_blob_name}_{self.current_date}.log"
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name,
        )

        # Probe the blob; create it only when it does not yet exist.
        try:
            blob_client.get_blob_properties()
        except Exception:
            try:
                blob_client.create_append_blob(
                    headers={
                        "Content-Type": "text/plain; charset=utf-8",
                        "Cache-Control": "no-store",
                    }
                )
            except ResourceExistsError:
                # Another process created the blob between our probe and create;
                # safe to proceed with the existing blob.
                pass

        return blob_client

    def emit(self, record):
        """Append a formatted log record to the active Azure blob.

        Rotates to a new dated blob automatically when the calendar date
        changes (i.e. at midnight). Falls back to printing to stderr on
        any storage error so the application is never silenced by a logging
        failure.

        Args:
            record: The :class:`logging.LogRecord` to write.
        """
        try:
            # Rotate the active blob when the date rolls over at midnight.
            today = date.today()
            if today != self.current_date:
                self.current_date = today
                self.blob_client = self._create_blob_client()

            # Format the record and ensure it ends with a newline before appending.
            msg = self.format(record)
            if not msg.endswith("\n"):
                msg += "\n"

            self.blob_client.append_block(msg.encode("utf-8"))

        except Exception as e:
            print(f"Error writing to blob: {e}")
