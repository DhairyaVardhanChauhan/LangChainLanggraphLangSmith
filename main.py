import random
import json
from fastmcp import FastMCP
import math

mcp = FastMCP()

# -------------------
# TOOLS
# -------------------
@mcp.tool()
def roll_dice():
    """Roll a dice and return a number between 1 and 6"""
    return random.randint(1, 6)

@mcp.tool()
def add_numbers(a: int, b: int):
    """Add two numbers"""
    return a + b

# -------------------
# MATH TOOLS
# -------------------
@mcp.tool()
def subtract_numbers(a: int, b: int):
    """Subtract b from a"""
    return a - b

@mcp.tool()
def multiply_numbers(a: int, b: int):
    """Multiply two numbers"""
    return a * b

@mcp.tool()
def divide_numbers(a: float, b: float):
    """Divide a by b"""
    if b == 0:
        return "Error: Division by zero"
    return a / b

@mcp.tool()
def power(a: float, b: float):
    """Compute a raised to the power b"""
    return a ** b

@mcp.tool()
def factorial(n: int):
    """Compute factorial of n"""
    if n < 0:
        return "Error: Negative factorial not defined"
    return math.factorial(n)

# -------------------
# RESOURCES
# -------------------
@mcp.resource("hello://world")
def hello_resource():
    return "Hello from MCP!"

print("Registering user resource...")
@mcp.resource("user://{name}")
def user_resource(name: str):
    return json.dumps({
        "user": name,
        "message": f"Hello {name}",
        "random": random.randint(1, 100)
    })

# -------------------
# RUN (MUST BE LAST)
# -------------------
def main():
    mcp.run()

if __name__ == "__main__":
    main()