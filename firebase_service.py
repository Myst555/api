"""Firebase Cloud Messaging helpers for the ReadSmart backend.

Wraps the Firebase Admin SDK so the rest of the service can send push
notifications without knowing the SDK's API.

Credentials are read from the FIREBASE_SERVICE_ACCOUNT environment variable,
which should point at a service-account JSON file. This keeps the secret out of
the source code and lets the same code run in local development and in
deployment. If the variable is absent the module still imports cleanly; sending
simply reports that it is unavailable instead of crashing the whole service at
startup.
"""

import logging
import os
from typing import Any

import firebase_admin
from firebase_admin import credentials, messaging

_LOG = logging.getLogger(__name__)

_service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT", "").strip()


def _initialize_app() -> bool:
    """Initialize the Firebase Admin app once. Returns False if unavailable."""
    if firebase_admin._apps:
        return True

    if not _service_account_path:
        _LOG.warning(
            "FIREBASE_SERVICE_ACCOUNT is not set; push notifications are disabled."
        )
        return False

    if not os.path.exists(_service_account_path):
        _LOG.error(
            "Service account file not found at %s; push notifications are disabled.",
            _service_account_path,
        )
        return False

    try:
        firebase_admin.initialize_app(credentials.Certificate(_service_account_path))
        return True
    except Exception:
        _LOG.exception("Failed to initialize Firebase Admin SDK")
        return False


def is_available() -> bool:
    """True when push notifications can be sent."""
    return _initialize_app()


def send_push_notification(
    token: str,
    title: str,
    body: str,
    data: dict | None = None,
) -> str:
    """Send a notification to a single device token."""
    if not _initialize_app():
        raise RuntimeError("Firebase Admin SDK is not available on this server.")

    message = messaging.Message(
        notification=messaging.Notification(title=title, body=body),
        data=data or {},
        token=token,
    )
    return messaging.send(message)


def send_topic_notification(
    topic: str,
    title: str,
    body: str,
    data: dict | None = None,
) -> str:
    """Send a notification to every device subscribed to a topic.

    Topic messaging is preferred over per-device sends: a teacher publishing one
    lesson should reach every enrolled student, and the client already subscribes
    to ``school_<schoolId>`` on sign-in. This also avoids a failure when a
    student's device token has gone stale.
    """
    if not _initialize_app():
        raise RuntimeError("Firebase Admin SDK is not available on this server.")

    message = messaging.Message(
        notification=messaging.Notification(title=title, body=body),
        data=data or {},
        topic=topic,
    )
    return messaging.send(message)


def send_multicast_notification(
    tokens: list[str],
    title: str,
    body: str,
    data: dict | None = None,
) -> dict[str, Any]:
    """Send a notification to many device tokens, reporting per-token results.

    FCM limits a single multicast call to 500 tokens, so the list is chunked.
    """
    if not _initialize_app():
        raise RuntimeError("Firebase Admin SDK is not available on this server.")

    results: dict[str, Any] = {
        "successCount": 0,
        "failureCount": 0,
        "responses": [],
    }

    for start in range(0, len(tokens), 500):
        chunk = tokens[start : start + 500]
        response = messaging.send_each_for_multicast(
            messaging.MulticastMessage(
                notification=messaging.Notification(title=title, body=body),
                data=data or {},
                tokens=chunk,
            )
        )
        results["successCount"] += response.success_count
        results["failureCount"] += response.failure_count
        results["responses"].extend(
            {"index": i, "success": r.success, "message_id": r.message_id}
            for i, r in enumerate(response.responses)
        )

    return results
