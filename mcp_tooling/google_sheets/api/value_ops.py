"""Value operations following Google's official Python examples"""
from googleapiclient.errors import HttpError
from typing import List, Any, Optional, Dict

class ValueOperations:
    """Value operations matching official documentation"""
    
    def __init__(self, service):
        self.service = service
    
    def get_values(self, spreadsheet_id: str, range_name: str) -> List[List[Any]]:
        """Get values - matches official example"""
        try:
            result = (
                self.service.spreadsheets()
                .values()
                .get(spreadsheetId=spreadsheet_id, range=range_name)
                .execute()
            )
            rows = result.get("values", [])
            print(f"{len(rows)} rows retrieved")
            return rows
        except HttpError as error:
            print(f"An error occurred: {error}")
            return []
    
    def batch_get_values(self, spreadsheet_id: str, 
                        range_names: List[str]) -> List[Dict[str, Any]]:
        """Batch get values - matches official example"""
        try:
            result = (
                self.service.spreadsheets()
                .values()
                .batchGet(spreadsheetId=spreadsheet_id, ranges=range_names)
                .execute()
            )
            ranges = result.get("valueRanges", [])
            print(f"{len(ranges)} ranges retrieved")
            return ranges
        except HttpError as error:
            print(f"An error occurred: {error}")
            return []
    
    def update_values(self, spreadsheet_id: str, range_name: str,
                     value_input_option: str, values: List[List[Any]]) -> Dict[str, Any]:
        """Update values - matches official example
        value_input_option: 'RAW' or 'USER_ENTERED'
        """
        try:
            body = {"values": values}
            result = (
                self.service.spreadsheets()
                .values()
                .update(
                    spreadsheetId=spreadsheet_id,
                    range=range_name,
                    valueInputOption=value_input_option,
                    body=body,
                )
                .execute()
            )
            print(f"{result.get('updatedCells')} cells updated.")
            return result
        except HttpError as error:
            print(f"An error occurred: {error}")
            raise error
    
    def batch_update_values(self, spreadsheet_id: str, 
                           data: List[Dict[str, Any]],
                           value_input_option: str = 'USER_ENTERED') -> Dict[str, Any]:
        """Batch update values - enhanced from official example
        data: List of {'range': str, 'values': List[List[Any]]}
        """
        try:
            body = {
                "valueInputOption": value_input_option, 
                "data": [
                    {
                        "range": item['range'],
                        "values": item['values'],
                        "majorDimension": item.get('majorDimension', 'ROWS')
                    }
                    for item in data
                ]
            }
            result = (
                self.service.spreadsheets()
                .values()
                .batchUpdate(spreadsheetId=spreadsheet_id, body=body)
                .execute()
            )
            print(f"{result.get('totalUpdatedCells')} cells updated.")
            return result
        except HttpError as error:
            print(f"An error occurred: {error}")
            raise error
    
    def append_values(self, spreadsheet_id: str, range_name: str,
                     value_input_option: str, values: List[List[Any]]) -> Dict[str, Any]:
        """Append values - based on official documentation"""
        try:
            body = {"values": values}
            result = (
                self.service.spreadsheets()
                .values()
                .append(
                    spreadsheetId=spreadsheet_id,
                    range=range_name,
                    valueInputOption=value_input_option,
                    body=body,
                )
                .execute()
            )
            print(
                f"{result.get('updates', {}).get('updatedCells', 0)} cells appended."
            )
            return result
        except HttpError as error:
            print(f"An error occurred: {error}")
            raise error
    
    def clear_values(self, spreadsheet_id: str, range_name: str) -> Dict[str, Any]:
        """Clear values from a range"""
        try:
            result = (
                self.service.spreadsheets()
                .values()
                .clear(
                    spreadsheetId=spreadsheet_id,
                    range=range_name
                )
                .execute()
            )
            print(f"Range {range_name} cleared.")
            return result
        except HttpError as error:
            print(f"An error occurred: {error}")
            raise error
    
    def batch_clear_values(self, spreadsheet_id: str, 
                          ranges: List[str]) -> Dict[str, Any]:
        """Clear multiple ranges in a single request"""
        try:
            body = {"ranges": ranges}
            result = (
                self.service.spreadsheets()
                .values()
                .batchClear(
                    spreadsheetId=spreadsheet_id,
                    body=body
                )
                .execute()
            )
            print(f"{len(ranges)} ranges cleared.")
            return result
        except HttpError as error:
            print(f"An error occurred: {error}")
            raise error
    
    def get_values_with_formatting(self, spreadsheet_id: str, 
                                  range_name: str,
                                  value_render_option: str = 'FORMATTED_VALUE',
                                  date_time_render_option: str = 'FORMATTED_STRING') -> Dict[str, Any]:
        """Get values with formatting options"""
        try:
            result = (
                self.service.spreadsheets()
                .values()
                .get(
                    spreadsheetId=spreadsheet_id,
                    range=range_name,
                    valueRenderOption=value_render_option,
                    dateTimeRenderOption=date_time_render_option
                )
                .execute()
            )
            return {
                'values': result.get('values', []),
                'range': result.get('range'),
                'majorDimension': result.get('majorDimension')
            }
        except HttpError as error:
            print(f"An error occurred: {error}")
            return {}