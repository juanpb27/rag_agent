from typing import Dict, List, Any


def get_driver_status(driver_id: str) -> Dict[str, Any]:
    """Get the current status of a driver."""
    return {
        "status": "available",
        "eta": "5 minutes"
    }


def calculate_eta(origin: str, destination: str) -> Dict[str, Any]:
    """Calculate estimated time of arrival between two locations."""
    return {
        "estimated_time": "15 minutes",
        "distance": "8.5 km"
    }


def create_support_ticket(user_email: str, message: str) -> Dict[str, Any]:
    """Create a support ticket for a user."""
    return {
        "ticket_id": "TK-2025-001",
        "status": "created"
    }


TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "get_driver_status",
        "description": "Get the current status and estimated time of arrival for a specific driver",
        "input_schema": {
            "type": "object",
            "properties": {
                "driver_id": {
                    "type": "string",
                    "description": "The unique identifier of the driver"
                }
            },
            "required": ["driver_id"]
        }
    },
    {
        "name": "calculate_eta",
        "description": "Calculate estimated time of arrival and distance between two locations",
        "input_schema": {
            "type": "object",
            "properties": {
                "origin": {
                    "type": "string",
                    "description": "The starting location or address"
                },
                "destination": {
                    "type": "string",
                    "description": "The destination location or address"
                }
            },
            "required": ["origin", "destination"]
        }
    },
    {
        "name": "create_support_ticket",
        "description": "Create a new support ticket for user assistance",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_email": {
                    "type": "string",
                    "description": "The email address of the user requesting support"
                },
                "message": {
                    "type": "string",
                    "description": "The support message or issue description"
                }
            },
            "required": ["user_email", "message"]
        }
    }
]


TOOL_FUNCTIONS_MAP: Dict[str, callable] = {
    "get_driver_status": get_driver_status,
    "calculate_eta": calculate_eta,
    "create_support_ticket": create_support_ticket
}
