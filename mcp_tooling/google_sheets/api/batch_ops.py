"""Batch operations following Google's recommendations"""
import time
from typing import List, Dict, Any
from collections import deque
from googleapiclient.errors import HttpError
import logging

class BatchOperations:
    """Batch operations with rate limiting per Google's guidelines"""
    
    # Google Sheets API limits: 100 requests per 100 seconds per user
    MAX_REQUESTS_PER_100_SECONDS = 100
    
    def __init__(self, service):
        self.service = service
        self.request_times = deque()
        self.logger = logging.getLogger(__name__)
    
    def _check_rate_limit(self):
        """Ensure we don't exceed rate limits"""
        now = time.time()
        # Remove requests older than 100 seconds
        while self.request_times and self.request_times[0] < now - 100:
            self.request_times.popleft()
        
        if len(self.request_times) >= self.MAX_REQUESTS_PER_100_SECONDS:
            # Wait until the oldest request is outside the window
            sleep_time = 100 - (now - self.request_times[0]) + 1
            self.logger.info(f"Rate limit reached, sleeping for {sleep_time} seconds")
            time.sleep(sleep_time)
    
    def batch_update(self, spreadsheet_id: str, requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute batch update following Google's patterns"""
        self._check_rate_limit()
        
        try:
            body = {"requests": requests}
            response = (
                self.service.spreadsheets()
                .batchUpdate(spreadsheetId=spreadsheet_id, body=body)
                .execute()
            )
            self.request_times.append(time.time())
            return response
        except HttpError as error:
            if error.resp.status == 429:
                # Rate limit exceeded, wait and retry
                self.logger.warning("Rate limit exceeded, waiting 60 seconds")
                time.sleep(60)
                return self.batch_update(spreadsheet_id, requests)
            raise error
    
    def optimize_requests(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize request order for efficiency"""
        # Group requests by type
        grouped = {}
        for request in requests:
            request_type = next(iter(request.keys()))
            if request_type not in grouped:
                grouped[request_type] = []
            grouped[request_type].append(request)
        
        # Order groups logically
        order = [
            'addSheet',  # Create sheets first
            'updateSpreadsheetProperties',  # Then update properties
            'updateSheetProperties',
            'updateCells',  # Then data
            'appendCells',
            'updateValues',
            'mergeCells',  # Then formatting
            'updateBorders',
            'addConditionalFormatRule',
            'addProtectedRange',  # Finally protection
        ]
        
        optimized = []
        for request_type in order:
            if request_type in grouped:
                optimized.extend(grouped[request_type])
                del grouped[request_type]
        
        # Add any remaining requests
        for requests_list in grouped.values():
            optimized.extend(requests_list)
        
        return optimized
    
    def execute_with_retry(self, spreadsheet_id: str, 
                          requests: List[Dict[str, Any]],
                          max_retries: int = 3) -> Dict[str, Any]:
        """Execute batch update with retry logic"""
        optimized_requests = self.optimize_requests(requests)
        
        for attempt in range(max_retries):
            try:
                return self.batch_update(spreadsheet_id, optimized_requests)
            except HttpError as e:
                if e.resp.status >= 500:  # Server error
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        self.logger.warning(f"Server error, retrying in {wait_time} seconds")
                        time.sleep(wait_time)
                        continue
                raise e
        
        raise Exception(f"Failed after {max_retries} attempts")