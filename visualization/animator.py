import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib.animation import FuncAnimation
import os 
from matplotlib.colors import LinearSegmentedColormap

class MazeAnimator:
    """
    Class for animating maze solving algorithms.
    """

    def __init__(self,maze, cell_size=20,interval=100,save_dir="output"):
        """
        Initialize the maze animator.
        
        Args:
            maze: The maze object to animate
            cell_size: Size of each cell in pixels
            interval: Time between frames in milliseconds
            save_dir: Directory to save animation frames/videos
        """

        self.maze=maze
        self.cell_size=cell_size
        self.interval=interval
        self.save_dir=save_dir

        os.makedirs(save_dir,exist_ok=True)

        self.fig,self.ax=plt.subplots(figsize= (maze.width * cell_size / 100, 
                                                maze.height * cell_size / 100))
        
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self.grid=np.ones((maze.height,maze.width,3))

        self._draw_maze()

        self.frames=[]

    def _draw_maze(self):
        """Draw the initial maze structure."""
        # Fill grid with appropriate colors for walls and paths
        for y in range(self.maze.height):
            for x in range(self.maze.width):
                if self.maze.is_wall(x, y):
                    self.grid[y, x] = [0.1, 0.1, 0.3]  # Dark blue for walls
                else:
                    self.grid[y, x] = [1.0, 1.0, 1.0]  # White for paths
        
        # Mark start and end
        if self.maze.start:
            start_x, start_y = self.maze.start
            self.grid[start_y, start_x] = [0.0, 0.8, 0.0]  # Green for start
        
        if self.maze.end:
            end_x, end_y = self.maze.end
            self.grid[end_y, end_x] = [0.8, 0.0, 0.0]  # Red for end
            
        # Display the maze
        self.image = self.ax.imshow(self.grid, interpolation='nearest')
        self.fig.tight_layout()

    def animate_solving(self, solver, save_as_video=True, filename="maze_solution"):
        """
        Animate the solving process.
        
        Args:
            solver: The solver object (BFS, DFS, etc.)
            save_as_video: Whether to save as video (True) or frames (False)
            filename: Base filename for the output
        """
        # Store initial state
        self.frames = [self._get_current_frame()]
        
        # Create a copy of the maze
        maze_copy = self.maze  # You might need a proper copy method
        
        # Set up tracking variables
        visited = set()
        path = []
        
        # Define update function for animation
        def update(frame_num):
            # This is a placeholder - actual implementation will 
            # depend on how your solvers record their steps
            if frame_num < len(solver.get_visited()):
                current = list(solver.get_visited())[frame_num]
                visited.add(current)
                
                # Update the grid to show the latest visited cell
                for y in range(self.maze.height):
                    for x in range(self.maze.width):
                        if (x, y) in visited and (x, y) != self.maze.start and (x, y) != self.maze.end:
                            # Lighter blue for visited cells
                            self.grid[y, x] = [0.7, 0.7, 1.0]
                            
                # Update the image data
                self.image.set_array(self.grid)
                
                # Capture this frame
                self.frames.append(self._get_current_frame())
            
            return [self.image]
        
        # Create the animation
        anim = FuncAnimation(
            self.fig, 
            update, 
            frames=len(solver.get_visited()) + 10,  # +10 for buffer
            interval=self.interval,
            blit=True
        )
        
        # Save the animation
        if save_as_video:
            self._save_as_video(anim, filename)
        else:
            self._save_as_frames(filename)
            
        return anim
    
    def _get_current_frame(self):
        """Capture current figure as a frame."""
        self.fig.canvas.draw()
        return np.array(self.fig.canvas.renderer.buffer_rgba())
        
    def _save_as_video(self, anim, filename):
        """Save animation as MP4 video."""
        mp4_file = os.path.join(self.save_dir, f"{filename}.mp4")
        anim.save(mp4_file, writer='ffmpeg', fps=15, dpi=100)
        print(f"Animation saved to {mp4_file}")
        
    def _save_as_frames(self, filename):
        """Save all frames as individual images."""
        for i, frame in enumerate(self.frames):
            frame_file = os.path.join(self.save_dir, f"{filename}_{i:03d}.png")
            plt.imsave(frame_file, frame)
        print(f"{len(self.frames)} frames saved to {self.save_dir}")

    def animate_path(self, path, save_as_video=True, filename="maze_path"):
        """
        Animate the final solution path.
        
        Args:
            path: List of (x,y) coordinates representing the solution path
            save_as_video: Whether to save as video
            filename: Base filename for output
        """
        # Reset frames
        self.frames = [self._get_current_frame()]
        
        # Define update function that reveals path step by step
        def update(frame_num):
            if frame_num < len(path):
                x, y = path[frame_num]
                if (x, y) != self.maze.start and (x, y) != self.maze.end:
                    # Yellow for path
                    self.grid[y, x] = [1.0, 0.9, 0.0]
                
                # Update the image
                self.image.set_array(self.grid)
                
                # Capture this frame
                self.frames.append(self._get_current_frame())
            
            return [self.image]
        
        # Create the animation
        anim = FuncAnimation(
            self.fig, 
            update, 
            frames=len(path) + 5,  # +5 for buffer
            interval=self.interval * 2,  # Slower for path visualization
            blit=True
        )
        
        # Save the animation
        if save_as_video:
            self._save_as_video(anim, filename)
        else:
            self._save_as_frames(filename)
            
        return anim
    
    def animate_solution(self, solver, save_as_video=True, filename="full_solution"):
        """
        Create a complete animation showing both exploration and final path.
        
        Args:
            solver: Solver object that has already solved the maze
            save_as_video: Whether to save as video
            filename: Base filename for output
        """
        # First run the solver if not already run
        if not solver.get_path():
            solver.solve(self.maze)
            
        # Animate the exploration process
        self.animate_solving(solver, save_as_video=False, filename=f"{filename}_explore")
        
        # Then show the final path
        return self.animate_path(solver.get_path(), save_as_video, filename)





