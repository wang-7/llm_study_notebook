from mcp.server.fastmcp import FastMCP

mcp = FastMCP('AddFunc')

@mcp.tool()
def add(a, b):
    return a+b

if __name__ == '__main__':
    mcp.run(transport='stdio')