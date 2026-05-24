import secrets

def generate_session_id(length_bytes=16):
    """
    Generate a cryptographically secure, URL-safe session ID.

    Args:
        length_bytes (int): Number of random bytes to use. 
                            Each byte adds ~1.3 characters to the output.

    Returns:
        str: A URL-safe session ID string.
    """
    if not isinstance(length_bytes, int) or length_bytes <= 0:
        raise ValueError("length_bytes must be a positive integer")
    
    return secrets.token_urlsafe(length_bytes)

if __name__ == "__main__":
    try:
        # Example: Generate a 16-byte (~22 char) session ID
        session_id = generate_session_id(16)
        print("Generated Session ID:", session_id)
    except Exception as e:
        print("Error generating session ID:", e)
