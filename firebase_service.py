import firebase_admin
from firebase_admin import credentials, messaging

if not firebase_admin._apps:
    cred = credentials.Certificate(
        "/home/ubuntu/api/firebase-service-account.json"
    )
    firebase_admin.initialize_app(cred)


def send_push_notification(
    token: str,
    title: str,
    body: str,
    data: dict | None = None,
):
    message = messaging.Message(
        notification=messaging.Notification(
            title=title,
            body=body,
        ),
        data=data or {},
        token=token,
    )

    return messaging.send(message)
