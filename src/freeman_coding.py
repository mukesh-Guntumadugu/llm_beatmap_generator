import numpy as np

def encode_freeman_8(points):
    """
    Encode a sequence of 2D points into an 8-directional Freeman Chain Code.
    
    Directions:
      3   2   1
       \  |  /
      4 - P - 0
       /  |  \
      5   6   7
      
    Args:
        points (list of tuple/list or np.ndarray): A list/array of (x, y) coordinates.
        Note: Assumes a Cartesian grid where y increases upwards. If using image
        coordinates (y increases downwards), you may want to reverse the y-axis.
        
    Returns:
        list of int: The Freeman chain code (values 0-7).
    """
    if len(points) < 2:
        return []
        
    chain_code = []
    points = np.array(points)
    
    # Calculate differences between consecutive points
    diffs = np.diff(points, axis=0)
    
    for dx, dy in diffs:
        # Standardize strictly to -1, 0, 1 based on the sign of the difference
        sx = np.sign(dx)
        sy = np.sign(dy)
        
        if sx == 1 and sy == 0:
            direction = 0
        elif sx == 1 and sy == 1:
            direction = 1
        elif sx == 0 and sy == 1:
            direction = 2
        elif sx == -1 and sy == 1:
            direction = 3
        elif sx == -1 and sy == 0:
            direction = 4
        elif sx == -1 and sy == -1:
            direction = 5
        elif sx == 0 and sy == -1:
            direction = 6
        elif sx == 1 and sy == -1:
            direction = 7
        else:
            # If sx == 0 and sy == 0, the point hasn't moved
            continue
            
        chain_code.append(direction)
        
    return chain_code

def decode_freeman_8(chain_code, start_point=(0, 0)):
    """
    Decode an 8-directional Freeman Chain Code back into a sequence of points.
    
    Args:
        chain_code (list of int): The Freeman chain code (values 0-7).
        start_point (tuple): The starting (x, y) coordinate.
        
    Returns:
        list of tuple: The sequence of (x, y) coordinates.
    """
    # Mapping directions to (dx, dy) based on the above encode mapping
    direction_map = {
        0: (1, 0),
        1: (1, 1),
        2: (0, 1),
        3: (-1, 1),
        4: (-1, 0),
        5: (-1, -1),
        6: (0, -1),
        7: (1, -1)
    }
    
    points = [tuple(start_point)]
    cx, cy = start_point
    
    for code in chain_code:
        dx, dy = direction_map.get(code, (0, 0))
        cx += dx
        cy += dy
        points.append((cx, cy))
        
    return points

# Example Usage:
if __name__ == "__main__":
    test_points = [(0, 0), (1, 0), (2, 1), (2, 2), (1, 3), (0, 3)]
    print(f"Original Points: {test_points}")
    encoded = encode_freeman_8(test_points)
    print(f"8-Directional Freeman Code: {encoded}")
    decoded = decode_freeman_8(encoded, start_point=(0, 0))
    print(f"Decoded Points: {decoded}")
