"""Error handling for Google Sheets API operations"""
from googleapiclient.errors import HttpError
from typing import Optional, Dict, Any, Callable
import logging
import time
from functools import wraps

class GoogleSheetsError(Exception):
    """Base exception for Google Sheets API errors"""
    pass

class RateLimitError(GoogleSheetsError):
    """Raised when API rate limit is exceeded"""
    pass

class AuthenticationError(GoogleSheetsError):
    """Raised when authentication fails"""
    pass

class PermissionError(GoogleSheetsError):
    """Raised when lacking permissions for an operation"""
    pass

class NotFoundError(GoogleSheetsError):
    """Raised when a resource is not found"""
    pass

class ErrorHandler:
    """Handles Google Sheets API errors with retry logic"""
    
    ERROR_MAPPING = {
        400: ("Bad Request - Invalid parameters", GoogleSheetsError),
        401: ("Unauthorized - Authentication failed", AuthenticationError),
        403: ("Forbidden - Insufficient permissions", PermissionError),
        404: ("Not Found - Spreadsheet or range not found", NotFoundError),
        429: ("Too Many Requests - Rate limit exceeded", RateLimitError),
        500: ("Internal Server Error", GoogleSheetsError),
        503: ("Service Unavailable", GoogleSheetsError)
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def handle_api_error(error: HttpError) -> None:
        """Convert HTTP errors to appropriate exceptions"""
        status = error.resp.status if hasattr(error, 'resp') else 500
        
        if status in ErrorHandler.ERROR_MAPPING:
            message, exception_class = ErrorHandler.ERROR_MAPPING[status]
            raise exception_class(f"{message}: {error}")
        else:
            raise GoogleSheetsError(f"Unknown error (status {status}): {error}")
    
    @staticmethod
    def retry_with_backoff(max_retries: int = 3, 
                          initial_delay: float = 1.0,
                          backoff_factor: float = 2.0,
                          retry_on: tuple = (429, 500, 503)):
        """Decorator for retrying operations with exponential backoff"""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                delay = initial_delay
                last_exception = None
                logger = logging.getLogger(__name__)
                
                for attempt in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except HttpError as e:
                        last_exception = e
                        status = e.resp.status if hasattr(e, 'resp') else 500
                        
                        if status in retry_on and attempt < max_retries - 1:
                            logger.info(
                                f"Attempt {attempt + 1}/{max_retries} failed with status {status}. "
                                f"Retrying in {delay} seconds..."
                            )
                            time.sleep(delay)
                            delay *= backoff_factor
                        else:
                            # Don't retry for client errors or last attempt
                            ErrorHandler.handle_api_error(e)
                    except Exception as e:
                        # Don't retry non-HTTP errors
                        raise
                
                # All retries exhausted
                if last_exception:
                    ErrorHandler.handle_api_error(last_exception)
                    
            return wrapper
        return decorator
    
    @staticmethod
    def safe_execute(func: Callable, *args, **kwargs) -> Optional[Any]:
        """Safely execute a function and return None on error"""
        try:
            return func(*args, **kwargs)
        except HttpError as e:
            logger = logging.getLogger(__name__)
            logger.error(f"API call failed: {e}")
            return None
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Unexpected error: {e}")
            return None
    
    @staticmethod
    def extract_error_details(error: HttpError) -> Dict[str, Any]:
        """Extract detailed information from an HTTP error"""
        details = {
            "status": None,
            "reason": None,
            "message": str(error),
            "details": None
        }
        
        if hasattr(error, 'resp'):
            details["status"] = error.resp.status
            details["reason"] = error.resp.reason
        
        if hasattr(error, 'content'):
            try:
                import json
                error_content = json.loads(error.content)
                details["details"] = error_content.get('error', {})
            except:
                pass
        
        return details
    
    @staticmethod
    def is_retryable_error(error: HttpError) -> bool:
        """Determine if an error is retryable"""
        if not hasattr(error, 'resp'):
            return False
        
        status = error.resp.status
        # Retry on rate limits and server errors
        return status in [429, 500, 502, 503, 504]
    
    @staticmethod
    def get_retry_after(error: HttpError) -> Optional[int]:
        """Extract Retry-After header from error response"""
        if not hasattr(error, 'resp'):
            return None
        
        retry_after = error.resp.get('Retry-After')
        if retry_after:
            try:
                return int(retry_after)
            except ValueError:
                pass
        
        # Default retry delays based on error type
        status = error.resp.status
        if status == 429:
            return 60  # Rate limit: wait 1 minute
        elif status >= 500:
            return 5   # Server error: wait 5 seconds
        
        return None