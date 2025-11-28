import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os
import json


class ScenarioGenerator:
    """Generate test scenarios with various grid configurations."""
    
    @staticmethod
    def save_scenario(name, map_array, description, output_dir="./test_scenarios"):
        """Save scenario to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save map
        np.save(os.path.join(output_dir, f"{name}_map.npy"), map_array)
        
        # Save metadata
        metadata = {
            'name': name,
            'description': description,
            'width': map_array.shape[1],
            'height': map_array.shape[0]
        }
        with open(os.path.join(output_dir, f"{name}_info.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Visualize
        ScenarioGenerator.visualize_map(map_array, name, description,
                                       os.path.join(output_dir, f"{name}.png"))
    
    @staticmethod
    def visualize_map(map_array, name, description, save_path):
        """Visualize and save map."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        cmap = colors.ListedColormap(['#ffffff', '#444444', '#ffcc99'])
        bounds = [0, 1, 2, 3]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        ax.imshow(map_array, origin='lower', cmap=cmap, norm=norm)
        ax.set_title(f"{name}\n{description}", fontsize=12, pad=15)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#ffffff', edgecolor='k', label='Free'),
            Patch(facecolor='#444444', edgecolor='k', label='Hard'),
            Patch(facecolor='#ffcc99', edgecolor='k', label='Soft')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # Scenario Generators
    # ========================================================================
    
    @staticmethod
    def empty_room(width=20, height=20):
        """Empty room - baseline."""
        map_array = np.zeros((height, width), dtype=np.int8)
        return "empty_room", map_array, "Empty room baseline"
    
    @staticmethod
    def single_soft_center(width=20, height=20):
        """Single soft obstacle in center."""
        map_array = np.zeros((height, width), dtype=np.int8)
        cx, cy = width // 2, height // 2
        map_array[cy-1:cy+2, cx-1:cx+2] = 2
        return "single_soft_center", map_array, "Single 3x3 soft obstacle in center"
    
    @staticmethod
    def corridor_narrow(width=20, height=20):
        """Narrow corridor with soft obstacle."""
        map_array = np.zeros((height, width), dtype=np.int8)
        
        # Walls
        map_array[0, :] = 1
        map_array[height-1, :] = 1
        map_array[:, 0] = 1
        map_array[:, width-1] = 1
        
        # Vertical walls to create corridor
        for y in range(3, height-3):
            map_array[y, width//3] = 1
            map_array[y, 2*width//3] = 1
        
        # Gaps
        map_array[height//3:height//3+3, width//3] = 0
        map_array[2*height//3:2*height//3+3, 2*width//3] = 0
        
        # Soft in middle
        map_array[height//2-1:height//2+2, width//2-1:width//2+2] = 2
        
        return "corridor_narrow", map_array, "Narrow corridor with blocking soft obstacle"
    
    @staticmethod
    def large_soft_region(width=20, height=20):
        """Large connected soft region."""
        map_array = np.zeros((height, width), dtype=np.int8)
        cx, cy = width // 2, height // 2
        
        # Create large irregular soft region
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                if 0 <= cy + dy < height and 0 <= cx + dx < width:
                    if abs(dx) + abs(dy) <= 7:
                        map_array[cy + dy, cx + dx] = 2
        
        return "large_soft_region", map_array, "Large connected soft region (high risk)"
    
    @staticmethod
    def scattered_soft(width=20, height=20, n_obstacles=12, seed=42):
        """Randomly scattered soft obstacles."""
        np.random.seed(seed)
        map_array = np.zeros((height, width), dtype=np.int8)
        
        for _ in range(n_obstacles):
            x = np.random.randint(2, width-3)
            y = np.random.randint(2, height-3)
            size = np.random.randint(1, 3)
            map_array[y-size:y+size+1, x-size:x+size+1] = 2
        
        return f"scattered_soft_seed{seed}", map_array, f"Scattered soft obstacles (n={n_obstacles})"
    
    @staticmethod
    def mixed_obstacles(width=20, height=20, seed=42):
        """Mix of hard and soft obstacles."""
        np.random.seed(seed)
        map_array = np.zeros((height, width), dtype=np.int8)
        
        # Random hard (5%)
        hard_mask = np.random.rand(height, width) < 0.05
        map_array[hard_mask] = 1
        
        # Random soft (10%)
        soft_mask = np.random.rand(height, width) < 0.10
        soft_mask = soft_mask & (map_array == 0)
        map_array[soft_mask] = 2
        
        return f"mixed_obstacles_seed{seed}", map_array, "Mixed hard and soft obstacles"
    
    @staticmethod
    def four_rooms(width=20, height=20):
        """Four rooms connected by doorways."""
        map_array = np.zeros((height, width), dtype=np.int8)
        
        # Outer walls
        map_array[0, :] = 1
        map_array[height-1, :] = 1
        map_array[:, 0] = 1
        map_array[:, width-1] = 1
        
        # Cross walls
        map_array[height//2, :] = 1
        map_array[:, width//2] = 1
        
        # Doorways
        map_array[height//2, width//4] = 0
        map_array[height//2, 3*width//4] = 0
        map_array[height//4, width//2] = 0
        map_array[3*height//4, width//2] = 0
        
        # Soft obstacles in each room
        map_array[height//4-1:height//4+2, width//4-1:width//4+2] = 2
        map_array[3*height//4-1:3*height//4+2, width//4-1:width//4+2] = 2
        map_array[height//4-1:height//4+2, 3*width//4-1:3*width//4+2] = 2
        map_array[3*height//4-1:3*height//4+2, 3*width//4-1:3*width//4+2] = 2
        
        return "four_rooms", map_array, "Four rooms with doorways and soft obstacles"
    
    @staticmethod
    def gauntlet(width=30, height=10):
        """Long corridor with multiple soft obstacles."""
        map_array = np.zeros((height, width), dtype=np.int8)
        
        # Walls
        map_array[0, :] = 1
        map_array[height-1, :] = 1
        
        # Soft obstacles every few cells
        for x in range(4, width-4, 5):
            map_array[height//2-1:height//2+2, x:x+2] = 2
        
        return "gauntlet", map_array, "Gauntlet with multiple soft obstacles"
    
    @staticmethod
    def checkerboard_soft(width=20, height=20):
        """Checkerboard pattern of soft regions."""
        map_array = np.zeros((height, width), dtype=np.int8)
        
        for y in range(2, height-2, 4):
            for x in range(2, width-2, 4):
                map_array[y:y+2, x:x+2] = 2
        
        return "checkerboard_soft", map_array, "Checkerboard pattern of soft obstacles"
    
    @staticmethod
    def soft_spiral(width=21, height=21):
        """Spiral pattern of soft obstacles."""
        map_array = np.zeros((height, width), dtype=np.int8)
        
        cx, cy = width // 2, height // 2
        angle = 0
        radius = 1
        
        for _ in range(80):
            x = int(cx + radius * np.cos(angle))
            y = int(cy + radius * np.sin(angle))
            
            if 0 <= x < width and 0 <= y < height:
                map_array[y, x] = 2
            
            angle += 0.4
            radius += 0.15
            
            if radius > min(width, height) // 2:
                break
        
        return "soft_spiral", map_array, "Spiral pattern (pathological case)"
    
    @staticmethod
    def start_stuck(width=15, height=15):
        """Start position inside soft region - must escape."""
        map_array = np.zeros((height, width), dtype=np.int8)
        
        cx, cy = width // 2, height // 2
        
        # Large soft region
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                if 0 <= cy + dy < height and 0 <= cx + dx < width:
                    if abs(dx) + abs(dy) <= 7:
                        map_array[cy + dy, cx + dx] = 2
        
        # Clear one escape path
        map_array[cy, :cx-2] = 0
        
        return "start_stuck", map_array, "Start inside soft region - escape challenge"
    
    @staticmethod
    def soft_maze(width=20, height=20):
        """Maze-like structure with soft obstacles."""
        map_array = np.zeros((height, width), dtype=np.int8)
        
        # Create maze walls (hard)
        for y in range(0, height, 4):
            map_array[y, :] = 1
        for x in range(0, width, 4):
            map_array[:, x] = 1
        
        # Clear some paths
        for y in range(2, height, 4):
            for x in range(2, width, 4):
                map_array[y, x] = 0
        
        # Add soft obstacles in open areas
        for y in range(1, height, 4):
            for x in range(1, width, 4):
                if map_array[y, x] == 0:
                    map_array[y, x] = 2
        
        return "soft_maze", map_array, "Maze with soft obstacles in corridors"


def generate_all_scenarios():
    """Generate all test scenarios."""
    print("Generating test scenarios...")
    
    output_dir = "./test_scenarios"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate each scenario
    scenarios = [
        ScenarioGenerator.empty_room(),
        ScenarioGenerator.single_soft_center(),
        ScenarioGenerator.corridor_narrow(),
        ScenarioGenerator.large_soft_region(),
        ScenarioGenerator.scattered_soft(seed=42),
        ScenarioGenerator.scattered_soft(seed=123),
        ScenarioGenerator.mixed_obstacles(seed=42),
        ScenarioGenerator.mixed_obstacles(seed=999),
        ScenarioGenerator.four_rooms(),
        ScenarioGenerator.gauntlet(),
        ScenarioGenerator.checkerboard_soft(),
        ScenarioGenerator.soft_spiral(),
        ScenarioGenerator.start_stuck(),
        ScenarioGenerator.soft_maze(),
    ]
    
    # Save all scenarios
    for name, map_array, description in scenarios:
        ScenarioGenerator.save_scenario(name, map_array, description, output_dir)
        print(f"  ✓ {name}: {description}")
    
    # Create summary
    summary = {
        'total_scenarios': len(scenarios),
        'scenarios': [
            {'name': name, 'description': desc, 'shape': list(map_array.shape)}
            for name, map_array, desc in scenarios
        ]
    }
    
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Generated {len(scenarios)} scenarios")
    print(f"✓ Saved to: {output_dir}")
    print(f"✓ Summary: {output_dir}/summary.json")


def load_scenario(name, scenarios_dir="./test_scenarios"):
    """Load a scenario from disk."""
    map_array = np.load(os.path.join(scenarios_dir, f"{name}_map.npy"))
    
    with open(os.path.join(scenarios_dir, f"{name}_info.json"), 'r') as f:
        info = json.load(f)
    
    return map_array, info


if __name__ == "__main__":
    generate_all_scenarios()
    
    print("\n" + "="*60)
    print("To use these scenarios:")
    print("="*60)
    print("""
    from generate_test_scenarios import load_scenario
    from setup import RoombaSoftPOMDPEnv
    
    # Load a scenario
    map_array, info = load_scenario("corridor_narrow")
    
    # Create environment with this map
    env = RoombaSoftPOMDPEnv(
        width=info['width'],
        height=info['height'],
        map_array=map_array
    )
    
    # Run your algorithm
    obs = env.reset()
    # ... your code here
    """)