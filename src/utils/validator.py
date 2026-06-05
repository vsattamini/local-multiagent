import ast
import logging

logger = logging.getLogger(__name__)

def validate_python_code(code: str) -> dict:
    """
    Validates Python code syntax using the ast module.
    Returns a dictionary with 'valid' (bool) and 'error' (str or None).
    """
    try:
        if not code.strip():
            return {"valid": False, "error": "Empty code block"}
            
        # Strip markdown code fencing if present
        clean_code = code
        if "```" in code:
            lines = code.splitlines()
            # aggressive check for markdown blocks
            content = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    content.append(line)
            
            # If we found blocks, use them. If not (maybe just one block ended), fallback to original or specific logic
            if content:
                clean_code = "\n".join(content)
            else:
                # Fallback: maybe the code was naked but had some ``` at the end? 
                # Or maybe it's just '```python\n...```'
                clean_code = code.replace("```python", "").replace("```", "")
        
        ast.parse(clean_code)
        return {"valid": True, "error": None}
    except SyntaxError as e:
        return {"valid": False, "error": f"SyntaxError at line {e.lineno}: {e.msg}"}
    except Exception as e:
        return {"valid": False, "error": f"Validation Error: {str(e)}"}
