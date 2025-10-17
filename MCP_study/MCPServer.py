from typing import Any
from click import format_filename
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP('weather')

NWS_API_BASE = 'https://api.weather.gov'
USER_AGENT = 'weather-app/1.0'

async def make_new_request(url: str) -> dict[str, Any] | None:
    headers = {
        'User-Agent': USER_AGENT,
        'Accept': 'application/geo+json'
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None
        
def format_alert(feature: dict) -> str:
    props = feature['properties']
    return f'''
事件：{props.get('event', 'Unknown')}
区域：{props.get('areaDesc', 'Unknown')}
严重性：{props.get('severity', 'Unknown')}
描述：{props.get('description', 'No Description available')}
指示：{props.get('instruction', 'No specific instruction provided')}
'''

@mcp.tool()
async def get_alert(state:str) -> str:
    url = f'{NWS_API_BASE}/alerts/active/area/{state}'
    data = await make_new_request(url)
    if not data or 'features' not in data:
        return '无法获取警报或未找到警报'
    if not data['features']:
        return '该州没有活跃警报'
    alerts = [format_alert(alert) for alert in data['features']]
    return '\n---\n'.join(alerts)
@mcp.tool()
async def get_forecast(latitude:float, longitude:float) -> str:
    points_url = f'{NWS_API_BASE}/points/{latitude},{longitude}'
    points_data = await make_new_request(points_url)

    if not points_data:
        return '无法获得此位置的预报数据'
        
    forecast_url = points_data['properties']['forecast']
    forecast_data = await make_new_request(forecast_url)

    if not forecast_data:
        return '无法获取此位置的详细预报'
    
    periods = forecast_data['properties']['peirods']
    forecasts = []
    for period in periods[:5]:
        forecast = f"""
{period['name']}:
温度：{period['temperature']}{period['temeratureUnit']}
风：{period['windSpeed']}{period['windDirection']}
预报：{period['detailedForcast']}
"""
        forecasts.append(forecast)
    return '\n---\n'.join(forecasts)

if __name__ == '__main__':
    mcp.run(transport='stdio')