"""Google Sheets API rate limiting configuration"""

# Google Sheets API quotas and limits
# Source: https://developers.google.com/sheets/api/limits

# Read requests
READ_REQUESTS_PER_100_SECONDS = 100
READ_REQUESTS_PER_DAY = 100_000_000

# Write requests  
WRITE_REQUESTS_PER_100_SECONDS = 100
WRITE_REQUESTS_PER_DAY = 100_000_000

# Overall API usage
API_REQUESTS_PER_100_SECONDS = 100
API_REQUESTS_PER_DAY = 100_000_000

# Batch operation limits
MAX_BATCH_UPDATE_REQUESTS = 1000  # Maximum requests per batchUpdate call
MAX_BATCH_GET_RANGES = 1000       # Maximum ranges per batchGet call

# Cell limits
MAX_CELLS_PER_REQUEST = 10_000_000  # Maximum cells that can be updated at once
MAX_COLUMNS_PER_SHEET = 18_278      # Maximum columns (A to ZZZ)
MAX_ROWS_PER_SHEET = 10_000_000     # Maximum rows per sheet

# Retry configuration
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_INITIAL_DELAY = 1.0  # seconds
DEFAULT_BACKOFF_FACTOR = 2.0
DEFAULT_MAX_DELAY = 60.0     # seconds

# Rate limit error codes
RATE_LIMIT_ERROR_CODES = [429, 503]

# Configuration for different usage patterns
USAGE_PATTERNS = {
    'light': {
        'batch_size': 10,
        'delay_between_batches': 1.0,
        'max_concurrent_requests': 5
    },
    'moderate': {
        'batch_size': 50,
        'delay_between_batches': 0.5,
        'max_concurrent_requests': 10
    },
    'heavy': {
        'batch_size': 100,
        'delay_between_batches': 1.0,
        'max_concurrent_requests': 5
    }
}