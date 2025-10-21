import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import librosa
import sounddevice as sd
import threading
import time
from pathlib import Path
import os

class TkinterSongLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Labeling Tool")
        self.root.geometry("1200x800")
        
        # Audio data
        self.y = None
        self.sr = None
        self.duration = 0
        self.audio_file = None
        self.labels_per_second = 10
        self.n_labels = 0
        
        # Labels for both types
        self.speed_labels = None
        self.pattern_labels = None
        self.current_label_set = "speed"  # Current active label set

        # Plateau selection state
        self.selected_plateau = None  # (start_idx, end_idx, label_value)
        self.plateau_highlight = None  # matplotlib patch for highlighting

        # Plateau Dividers - Updated for multiple dividers
        self.dividers = []  # List of divider indices
        self.selected_divider = None  # Index of currently selected divider in self.dividers list
        self.divider_lines = []  # Visual divider lines on plot
        self.divider_colors = ['orange', 'purple', 'cyan', 'magenta', 'yellow', 'lime', 'pink', 'brown']
                
        # Playback state
        self.is_playing = False
        self.position = 0.0
        self.current_label = 0
        self.play_start_time = 0
        self.play_start_pos = 0
        self.play_thread = None
        self.should_stop_playback = False

        self.auto_apply = True
        
        # GUI elements
        self.position_line = None
        self.label_line = None
        self.status_label = None
        self.update_timer = None
        
        self.setup_gui()
        self.setup_bindings()
        # Bind focus reset to any widget interaction
        self.root.bind_all('<Button-1>', lambda e: self.root.after_idle(self.reset_focus))
        
    def setup_gui(self):
        """Setup the GUI layout"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="5")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        ttk.Button(file_frame, text="Load Audio File", command=self.load_audio_file).grid(row=0, column=0, padx=(0, 10))
        self.file_label = ttk.Label(file_frame, text="No file loaded")
        self.file_label.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        # Label type selection
        label_type_frame = ttk.LabelFrame(file_frame, text="Current Label Set")
        label_type_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.label_type_var = tk.StringVar(value="speed")
        ttk.Radiobutton(label_type_frame, text="Speed Labels (0-9)", variable=self.label_type_var, 
                       value="speed", command=self.on_label_type_change).grid(row=0, column=0, padx=(0, 20))
        ttk.Radiobutton(label_type_frame, text="Pattern Labels (0-7)", variable=self.label_type_var, 
                       value="pattern", command=self.on_label_type_change).grid(row=0, column=1)
        
        # Copy buttons frame
        copy_frame = ttk.Frame(label_type_frame)
        copy_frame.grid(row=1, column=0, columnspan=2, pady=(10, 0))

        self.copy_speed_button = ttk.Button(copy_frame, text="Copy Speed → Pattern", 
                                        command=self.copy_speed_to_pattern, state="disabled")
        self.copy_speed_button.grid(row=0, column=0, padx=(0, 10))

        self.copy_pattern_button = ttk.Button(copy_frame, text="Copy Pattern → Speed", 
                                            command=self.copy_pattern_to_speed, state="disabled")
        self.copy_pattern_button.grid(row=0, column=1)
        
        # Control frame
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="5")
        control_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Playback controls
        playback_frame = ttk.Frame(control_frame)
        playback_frame.grid(row=0, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.play_button = ttk.Button(playback_frame, text="Play", command=self.toggle_play, state="disabled")
        self.play_button.grid(row=0, column=0, padx=(0, 10))
        
        ttk.Label(playback_frame, text="Position:").grid(row=0, column=1, padx=(0, 5))
        self.position_var = tk.DoubleVar()
        self.position_scale = ttk.Scale(playback_frame, from_=0, to=100, variable=self.position_var, 
                                       command=self.on_position_change, state="disabled")
        self.position_scale.grid(row=0, column=2, sticky=(tk.W, tk.E), padx=(0, 10))
        playback_frame.columnconfigure(2, weight=1)
        
        self.position_label = ttk.Label(playback_frame, text="0.0s / 0.0s")
        self.position_label.grid(row=0, column=3)
        
        # Label controls
        label_control_frame = ttk.Frame(control_frame)
        label_control_frame.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E))
        
        ttk.Label(label_control_frame, text="Current Label:").grid(row=0, column=0, padx=(0, 5))
        self.current_label_var = tk.IntVar()
        self.label_spinbox = ttk.Spinbox(label_control_frame, from_=0, to=9, width=5, 
                                        textvariable=self.current_label_var, command=self.on_label_change)
        self.label_spinbox.grid(row=0, column=1, padx=(0, 10))
        
        ttk.Button(label_control_frame, text="Apply Label", command=self.apply_label).grid(row=0, column=2, padx=(0, 10))
        ttk.Button(label_control_frame, text="Save Labels", command=self.save_labels).grid(row=0, column=3)

        quick_label_frame = ttk.LabelFrame(control_frame, text="Quick Labels", padding="5")
        quick_label_frame.grid(row=3, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(10, 0))

        # Create label buttons
        label_buttons = [
            (0, "0: None"),
            (1, "1: Vocals"), 
            (2, "2: Ambient"),
            (3, "3: Buildup"),
            (4, "4: Buildup2"),
            (5, "5: Pre-Drop"),
            (6, "6: Drop"),
            (7, "7: Drop2"),
            (8, "8: Hold")
        ]

        for i, (value, text) in enumerate(label_buttons):
            btn = ttk.Button(quick_label_frame, text=text, 
                            command=lambda v=value: self.set_quick_label(v))
            btn.grid(row=0, column=i, padx=2)

        # Auto-apply toggle
        auto_apply_frame = ttk.Frame(quick_label_frame)
        auto_apply_frame.grid(row=1, column=0, columnspan=7, pady=(10, 0))

        self.auto_apply_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(auto_apply_frame, text="Toggle Label Apply", 
                    variable=self.auto_apply_var, command=self.toggle_auto_apply).pack()
        
        # Plot frame
        plot_frame = ttk.LabelFrame(main_frame, text="Visualization", padding="5")
        plot_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 6), dpi=100)
        self.ax1, self.ax2 = self.fig.subplots(2, 1)
        
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Divider controls frame
        divider_frame = ttk.LabelFrame(control_frame, text="Divider Controls", padding="5")
        divider_frame.grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(10, 0))
        ttk.Button(divider_frame, text="Reset Dividers", command=self.reset_all_dividers).grid(row=0, column=1, padx=(0, 10))
                
        # Status bar
        self.status_label = ttk.Label(main_frame, text="Ready - Load an audio file to begin")
        self.status_label.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Instructions
        instructions = """
        Controls:
        • SPACE: Play/Pause 
        • LEFT/RIGHT ARROWS: Skip backward/forward 0.2s (hold Shift for 3s) 
        • 0-9: Set label (Speed mode) / 0-7: Set label (Pattern mode)
        • Left click on plot: Seek to position 
        • Right click on labels plot: Select plateau for editing 
        • D: Deselect current plateau or current divider
        • K: Delete current divider
        • R: Reset all dividers
        • I: Insert divider at current position 
        • ESC: Save labels 
        • Q: Quit
        """
        
        instructions_label = ttk.Label(main_frame, text=instructions, justify=tk.LEFT, 
                                     font=('TkDefaultFont', 8))
        instructions_label.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))

    def reset_focus(self):
        """Reset focus to main window"""
        self.root.focus_set()
    
    def setup_bindings(self):
        """Setup keyboard bindings"""
        self.root.bind('<KeyPress>', self.on_key_press)
        self.root.focus_set()  # Ensure window can receive key events
        
        # Canvas click binding
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.canvas.mpl_connect('button_press_event', self.on_divider_click)
        
    
    def load_audio_file(self):
        """Load audio file dialog"""
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.m4a *.aac"),
                ("WAV files", "*.wav"),
                ("MP3 files", "*.mp3"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.load_audio(file_path)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load audio file:\n{str(e)}")
    
    def load_audio(self, file_path):
        """Load and process audio file"""
        self.status_label.config(text="Loading audio...")
        self.root.update()
        
        # Load audio
        self.y, self.sr = librosa.load(file_path)
        self.duration = len(self.y) / self.sr
        self.audio_file = file_path
        
        # Setup labels
        self.n_labels = int(self.duration * self.labels_per_second)
        self.speed_labels = np.zeros(self.n_labels, dtype=int)
        self.pattern_labels = np.zeros(self.n_labels, dtype=int)
        
        # Check if existing labels file exists and load them
        self.load_existing_labels()
        
        # Reset playback state
        self.stop_playback()
        self.is_playing = False
        self.position = 0.0
        self.current_label = 0
        self.current_label_var.set(0)
        
        # Update GUI
        self.file_label.config(text=f"Loaded: {Path(file_path).name}")
        self.position_scale.config(to=self.duration, state="normal")
        self.play_button.config(state="normal")
        
        # Update label spinbox range based on current type
        self.on_label_type_change()
        
        # Setup plot
        self.setup_plot()
        
        # Start update timer
        self.start_update_timer()

        self.update_copy_button_states()
        
        self.status_label.config(text=f"Ready! {self.duration:.1f}s audio, {self.n_labels} labels")
    
    def load_existing_labels(self):
        """Load existing labels if the npz file exists"""
        output_path = Path("labels") / Path(self.audio_file).with_suffix('.labels.npz').name
        
        if output_path.exists():
            try:
                data = np.load(output_path)
                if 'speed_labels' in data:
                    self.speed_labels = data['speed_labels']
                if 'pattern_labels' in data:
                    self.pattern_labels = data['pattern_labels']
                print(f"Loaded existing labels from {output_path}")
            except Exception as e:
                print(f"Error loading existing labels: {e}")
    
    def on_label_type_change(self):
        """Handle label type change"""
        self.current_label_set = self.label_type_var.get()
        max_label = 9 if self.current_label_set == "speed" else 8
        self.label_spinbox.config(to=max_label)
        if self.current_label_var.get() > max_label:
            self.current_label_var.set(max_label)
            self.current_label = max_label
        if hasattr(self, 'ax2'):
            self.update_plot_labels()

        self.update_copy_button_states()

    def set_quick_label(self, label_value):
        """Set current label from quick buttons"""
        max_label = 9 if self.current_label_set == "speed" else 8
        if label_value <= max_label:
            self.current_label = label_value
            self.current_label_var.set(label_value)
            if self.auto_apply:
                self.apply_label()
    
    def get_current_labels(self):
        """Get the currently active label array"""
        return self.speed_labels if self.current_label_set == "speed" else self.pattern_labels
    
    def setup_plot(self):
        """Setup the matplotlib plots"""
        self.ax1.clear()
        self.ax2.clear()
        
        # Downsample waveform for display
        step = max(1, len(self.y) // 2000)
        times = np.linspace(0, self.duration, len(self.y[::step]))
        
        # Waveform plot
        self.ax1.plot(times, self.y[::step], 'b-', alpha=0.6, linewidth=0.5)
        self.position_line = self.ax1.axvline(0, color='red', linewidth=2)
        self.ax1.set_ylabel('Waveform')
        self.ax1.grid(True, alpha=0.3)
        
        # Labels plot
        label_times = np.linspace(0, self.duration, self.n_labels)
        current_labels = self.get_current_labels()
        self.label_line, = self.ax2.plot(label_times, current_labels, 'g-', linewidth=2)
        self.position_line_labels = self.ax2.axvline(0, color='red', linewidth=2)  # vertical marker

        self.plateau_highlight = None
        
        label_type_str = "Speed" if self.current_label_set == "speed" else "Pattern"
        self.ax2.set_ylabel(f'{label_type_str} Labels')
        self.ax2.set_xlabel('Time (s)')
        
        y_max = 9.5 if self.current_label_set == "speed" else 8.5
        self.ax2.set_ylim(0, y_max)
        self.ax2.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def update_plot_labels(self):
        """Update plot with current label type"""
        if self.ax2 and self.label_line:
            label_type_str = "Speed" if self.current_label_set == "speed" else "Pattern"
            self.ax2.set_ylabel(f'{label_type_str} Labels')
            y_max = 9.5 if self.current_label_set == "speed" else 8.5
            self.ax2.set_ylim(0, y_max)
            
            # Update the line data
            current_labels = self.get_current_labels()
            self.label_line.set_ydata(current_labels)
            self.canvas.draw()
    
    def start_update_timer(self):
        """Start the update timer for 10fps updates"""
        if self.update_timer:
            self.root.after_cancel(self.update_timer)
        self.update_display()

    def skip_time(self, delta_seconds):
        """Skip forward/backward by delta_seconds, applying labels if auto_apply is on"""
        if not self.audio_file:
            return
            
        old_position = self.position
        new_position = max(0, min(self.position + delta_seconds, self.duration))
        
        # If auto_apply is on, apply current label to the range we're skipping over
        if self.auto_apply and abs(delta_seconds) > 0.01:  # Only if meaningful skip
            start_time = min(old_position, new_position)
            end_time = max(old_position, new_position)
            self.apply_label_range(start_time, end_time)
        
        # Update position
        self.position = new_position
        
        # If we're playing and position changed significantly, restart playback
        if self.is_playing and abs(new_position - old_position) > 0.1:
            self.stop_playback()
            self.play_from_position()

    def on_divider_click(self, event):
        """Handle clicking on dividers to select them"""
        if event.inaxes == self.ax2 and event.xdata is not None and event.button == 1:  # Left click
            # Check if click is near any divider
            click_time = event.xdata
            closest_divider = None
            min_distance = float('inf')
            
            for i, divider_idx in enumerate(self.dividers):
                divider_time = divider_idx / self.labels_per_second
                distance = abs(click_time - divider_time)
                if distance < 0.5 and distance < min_distance:  # Within 0.5 second tolerance
                    min_distance = distance
                    closest_divider = i
            
            if closest_divider is not None:
                self.selected_divider = closest_divider
                self.update_selected_divider_display()
            else:
                # If not clicking on a divider, deselect
                self.selected_divider = None
                self.update_selected_divider_display()

    def update_selected_divider_display(self):
        """Update the display to show which divider is selected"""
        if self.selected_divider is not None:
            divider_idx = self.dividers[self.selected_divider]
            time_pos = divider_idx / self.labels_per_second
        
        # Update visual display
        self.update_divider_display()

    def move_selected_divider(self, direction):
        """Move the selected divider by one frame in the given direction"""
        if self.selected_divider is None or not self.audio_file:
            return
        
        # Calculate frame size (1/labels_per_second)
        frame_size = 1  # 1 label index = 1 frame
        
        # Get current divider index
        current_idx = self.dividers[self.selected_divider]
        new_idx = current_idx + (direction * frame_size)
        
        # Clamp to valid range
        new_idx = max(0, min(new_idx, self.n_labels - 1))
        
        # Update divider position
        self.dividers[self.selected_divider] = new_idx
        
        # Update audio position to the new divider location
        new_time = new_idx / self.labels_per_second
        self.position = new_time
        
        # If playing, restart from new position
        if self.is_playing:
            self.stop_playback()
            self.play_from_position()
        
        # Update displays
        self.update_selected_divider_display()
        self.update_divider_display()

    def delete_selected_divider(self):
        """Delete the currently selected divider"""
        if self.selected_divider is not None:
            # Remove the divider
            removed_idx = self.dividers.pop(self.selected_divider)
            print(f"Deleted divider at index {removed_idx}")

            # Clear selection
            self.selected_divider = None
            self.update_selected_divider_display()
            self.update_divider_display()
        else: 
            print("No divider selected")
    
    def update_display(self):
        """Update display periodically (10fps)"""
        if not self.audio_file:
            return
            
        # Update position if playing
        if self.is_playing and not self.should_stop_playback:
            elapsed = time.time() - self.play_start_time
            new_position = self.play_start_pos + elapsed
            
            # Check if we've reached the end
            if new_position >= self.duration:
                self.stop_playback()
                self.position = self.duration
            else:
                self.position = new_position
                
                # Auto-apply labels while playing
                if self.auto_apply:
                    self.apply_label()
        
        # Update GUI elements
        self.position_var.set(self.position)
        self.position_label.config(text=f"{self.position:.1f}s / {self.duration:.1f}s")
        
        # Update plot
        if self.position_line:
            self.position_line.set_xdata([self.position, self.position])
        if hasattr(self, 'position_line_labels') and self.position_line_labels: 
            self.position_line_labels.set_xdata([self.position, self.position])
        if self.label_line:
            current_labels = self.get_current_labels()
            self.label_line.set_ydata(current_labels)
        
        # Update status
        status_text = f"Label: {self.current_label} | Position: {self.position:.1f}s | "
        status_text += f"Playing: {self.is_playing} | Mode: {self.current_label_set.title()}"
        self.status_label.config(text=status_text)
        
        # Update play button text
        self.play_button.config(text="Pause" if self.is_playing else "Play")
        
        try:
            self.canvas.draw_idle()
        except:
            pass
        
        # Schedule next update (100ms = 10fps)
        self.update_timer = self.root.after(100, self.update_display)
    
    def on_key_press(self, event):
        """Handle keyboard input"""
        if not self.audio_file:
            return
            
        key = event.keysym
        
        if key == 'space':
            self.toggle_play()
        elif key == 'Escape':
            self.save_labels()
        elif key == 'q':
            self.root.quit()
        elif key.isdigit():
            digit = int(key)
            max_label = 9 if self.current_label_set == "speed" else 8
            if digit <= max_label:
                self.current_label = digit
                self.current_label_var.set(digit)
                if self.auto_apply:
                    self.apply_label()
        elif key == 'Left':
            if self.selected_divider is not None:
                # Move selected divider left
                self.move_selected_divider(-1)
            if event.state & 0x0001: 
                self.skip_time(-3) # skip 3 seconds
            else:
                self.skip_time(-0.2)  # Skip back 1/5 second
        elif key == 'Right':
            if self.selected_divider is not None:
                # Move selected divider right
                self.move_selected_divider(1)
            if event.state & 0x0001:
                self.skip_time(3)
            else:
                self.skip_time(0.2)    
        elif key == 'd':
            if self.selected_divider is not None:
                # Deselect divider
                self.selected_divider = None
                self.update_selected_divider_display()
            else:
                # Clear plateau selection (existing functionality)
                self.clear_plateau_selection()
        elif key == 'k':
            # Delete selected divider
            if self.selected_divider is not None:
                self.delete_selected_divider()
        elif key == 'i':
            self.insert_divider()
    
    def on_canvas_click(self, event):
        """Handle canvas click for seeking"""
        if event.inaxes == self.ax1 and event.xdata is not None:
            self.seek(event.xdata)
            
        elif event.inaxes == self.ax2 and event.xdata is not None:
        # Labels plot click - check for plateau selection
            if event.button == 3:  # Right click for plateau selection
                plateau = self.find_plateau_at_position(event.xdata)
                if plateau:
                    self.selected_plateau = plateau
                    start_idx, end_idx, label_value = plateau
                    self.highlight_plateau(start_idx, end_idx)
                    self.show_plateau_edit_dialog(plateau)
            else:  # Left click - seek as before
                self.seek(event.xdata)

    def find_plateau_at_position(self, time_pos):
        """Find the plateau (continuous same-value region) at the given time position, respecting dividers"""
        current_labels = self.get_current_labels()
        
        # Convert time to label index
        label_idx = int(time_pos * self.labels_per_second)
        if label_idx < 0 or label_idx >= len(current_labels):
            return None
        
        label_value = current_labels[label_idx]
        
        # Get sorted divider positions (sorted)
        dividers = sorted(self.dividers)
        
        # Find start of plateau (go backwards until value changes or hit a divider)
        start_idx = label_idx
        while start_idx > 0 and current_labels[start_idx - 1] == label_value:
            # Check if we hit a divider
            if start_idx in dividers:
                break
            start_idx -= 1
        
        # Find end of plateau (go forwards until value changes or hit a divider)
        end_idx = label_idx
        while end_idx < len(current_labels) - 1 and current_labels[end_idx + 1] == label_value:
            # Check if we hit a divider
            end_idx += 1 # move to the position first since we'ere checking the end
            if end_idx in dividers:
                break
        
        return (start_idx, end_idx, label_value)
    
    def highlight_plateau(self, start_idx, end_idx):
        """Highlight the selected plateau on the plot"""
        # Remove existing highlight
        if self.plateau_highlight:
            self.plateau_highlight.remove()
        
        # Convert indices to time
        start_time = start_idx / self.labels_per_second
        end_time = (end_idx + 1) / self.labels_per_second  # +1 to include the end point
        
        # Create highlight rectangle
        y_max = 9.5 if self.current_label_set == "speed" else 8.5
        self.plateau_highlight = self.ax2.axvspan(start_time, end_time, 
                                                alpha=0.3, color='yellow', 
                                                zorder=10)
        self.canvas.draw()

    def clear_plateau_selection(self):
        """Clear the current plateau selection and highlighting"""
        # Remove highlight
        if self.plateau_highlight:
            self.plateau_highlight.remove()
            self.plateau_highlight = None
            self.canvas.draw()
        
        # Clear selection
        self.selected_plateau = None

    def show_plateau_edit_dialog(self, plateau):
        """Show dialog to edit the selected plateau"""
        start_idx, end_idx, current_value = plateau
        
        # Create popup window
        popup = tk.Toplevel(self.root)
        popup.title("Edit Plateau")
        popup.geometry("300x150")
        popup.transient(self.root)
        popup.grab_set()
        
        # Center the popup
        popup.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, 
                                self.root.winfo_rooty() + 50))
        
        # Content
        ttk.Label(popup, text=f"Plateau: indices {start_idx}-{end_idx}").pack(pady=10)
        ttk.Label(popup, text=f"Current value: {current_value}").pack()
        ttk.Label(popup, text=f"Duration: {(end_idx - start_idx + 1) / self.labels_per_second:.2f}s").pack()
        
        # Entry for new value
        entry_frame = ttk.Frame(popup)
        entry_frame.pack(pady=10)
        ttk.Label(entry_frame, text="New value:").pack(side=tk.LEFT)
        
        new_value_var = tk.StringVar(value=str(current_value))
        entry = ttk.Entry(entry_frame, textvariable=new_value_var, width=10)
        entry.pack(side=tk.LEFT, padx=(5, 0))
        entry.focus()
        entry.select_range(0, tk.END)
        
        # Buttons
        button_frame = ttk.Frame(popup)
        button_frame.pack(pady=10)

        def apply_change():
            try:
                new_value = int(new_value_var.get())
                max_label = 9 if self.current_label_set == "speed" else 8
                if 0 <= new_value <= max_label:
                    self.apply_plateau_change(start_idx, end_idx, new_value)
                    popup.destroy()
                else:
                    messagebox.showerror("Error", f"Value must be between 0 and {max_label}")
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid integer")
        
        def cancel():
            # Remove highlight
            if self.plateau_highlight:
                self.plateau_highlight.remove()
                self.plateau_highlight = None
                self.canvas.draw()
            popup.destroy()
        
        ttk.Button(button_frame, text="Apply", command=apply_change).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Cancel", command=cancel).pack(side=tk.LEFT)
        
        # Bind Enter and Escape keys
        popup.bind('<Return>', lambda e: apply_change())
        popup.bind('<Escape>', lambda e: cancel())

    def reset_all_dividers(self):
        """Clear all dividers"""
        self.dividers.clear()
        self.selected_divider = None
        self.update_selected_divider_display()
        self.update_divider_display()

    def copy_speed_to_pattern(self):
        """Copy speed labels to pattern labels with confirmation"""
        if not self.confirm_copy_operation("speed labels", "pattern labels"):
            return
        
        # Copy speed labels to pattern labels
        self.pattern_labels = self.speed_labels.copy()
        
        # Clamp pattern labels to valid range (0-7)
        self.pattern_labels = np.clip(self.pattern_labels, 0, 8)
        
        # Update display
        self.update_plot_labels()
        self.update_copy_button_states()
        
        print("Copied speed labels to pattern labels (clamped to 0-8)")

    def copy_pattern_to_speed(self):
        """Copy pattern labels to speed labels with confirmation"""
        if not self.confirm_copy_operation("pattern labels", "speed labels"):
            return
        
        # Copy pattern labels to speed labels
        self.speed_labels = self.pattern_labels.copy()
        
        # Update display
        self.update_plot_labels()
        self.update_copy_button_states()
        
        print("Copied pattern labels to speed labels")

    def confirm_copy_operation(self, source_type, target_type):
        """Show confirmation dialog for copy operation"""
        message = f"This will overwrite all current {target_type} with {source_type}.\n\nAre you sure you want to continue?"
        
        result = messagebox.askyesno("Confirm Copy Operation", message, 
                                    icon='warning', default='no')
        return result
    
    def update_copy_button_states(self):
        """Update the enabled/disabled state of copy buttons based on current conditions"""
        if not hasattr(self, 'copy_speed_button') or not hasattr(self, 'copy_pattern_button'):
            return
        
        # Check if speed labels have non-zero values
        speed_has_nonzero = self.speed_labels is not None and np.any(self.speed_labels != 0)
        
        # Check if pattern labels have non-zero values  
        pattern_has_nonzero = self.pattern_labels is not None and np.any(self.pattern_labels != 0)
        
        # Copy Speed → Pattern button: enabled if speed has non-zero AND current set is pattern
        if speed_has_nonzero and self.current_label_set == "pattern":
            self.copy_speed_button.config(state="normal")
        else:
            self.copy_speed_button.config(state="disabled")
        
        # Copy Pattern → Speed button: enabled if pattern has non-zero AND current set is speed
        if pattern_has_nonzero and self.current_label_set == "speed":
            self.copy_pattern_button.config(state="normal")
        else:
            self.copy_pattern_button.config(state="disabled")

    def apply_plateau_change(self, start_idx, end_idx, new_value):
        """Apply the new value to the selected plateau"""
        current_labels = self.get_current_labels()
        current_labels[start_idx:end_idx + 1] = new_value

        self.clear_plateau_selection()
        
        # Update plot
        self.update_plot_labels()

        self.update_copy_button_states()
        
        print(f"Changed plateau indices {start_idx}-{end_idx} to value {new_value}")

    def insert_divider(self):
        """Insert a divider at the current position"""
        if not self.audio_file:
            return
            
        # Convert current position to label index
        label_idx = int(self.position * self.labels_per_second)
        if label_idx < 0 or label_idx >= self.n_labels:
            return
        
        # Check if divider already exists at this position
        if label_idx in self.dividers:
            print(f"Divider already exists at index {label_idx}")
            return
        
        # Add new divider
        self.dividers.append(label_idx)
        self.dividers.sort()  # Keep dividers sorted
        
        print(f"Inserted divider at index {label_idx} (time: {self.position:.2f}s)")
        self.update_divider_display()

    def update_divider_display(self):
        """Update divider lines on the plot"""
        # Remove existing divider lines
        for line in self.divider_lines:
            line.remove()
        self.divider_lines.clear()
        
        # Add divider lines
        for i, divider_idx in enumerate(self.dividers):
            time_pos = divider_idx / self.labels_per_second
            color = self.divider_colors[i % len(self.divider_colors)]
            
            # Make selected divider thicker and more opaque
            if i == self.selected_divider:
                line = self.ax2.axvline(time_pos, color=color, linewidth=2, 
                                    linestyle='-', alpha=1.0, zorder=20)
            else:
                line = self.ax2.axvline(time_pos, color=color, linewidth=1, 
                                    linestyle='-', alpha=0.8, zorder=15)
            
            self.divider_lines.append(line)
        
        self.canvas.draw()
    
    def on_position_change(self, value):
        """Handle position scale change"""
        if not self.is_playing:  # Only allow manual seeking when not playing
            self.seek(float(value))
    
    def on_label_change(self):
        """Handle label spinbox change"""
        self.current_label = self.current_label_var.get()
    
    def stop_playback(self):
        """Stop current playback completely"""
        self.should_stop_playback = True
        sd.stop()
        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.join(timeout=0.1)  # Brief wait for thread cleanup
        self.is_playing = False
        self.should_stop_playback = False
    
    def seek(self, time_pos):
        """Jump to position"""
        old_position = self.position
        self.position = max(0, min(time_pos, self.duration))
        
        # If we're playing and position changed significantly, restart playback
        if self.is_playing and abs(self.position - old_position) > 0.1:
            self.stop_playback()
            self.play_from_position()
    
    def toggle_play(self):
        """Play/pause toggle"""
        if self.is_playing:
            self.stop_playback()
        else:
            self.play_from_position()
    
    def play_from_position(self):
        """Start playback from current position"""
        # Stop any existing playback first
        self.stop_playback()
        
        start_sample = int(self.position * self.sr)
        audio_chunk = self.y[start_sample:]
        
        if len(audio_chunk) > 100:  # Minimum chunk size
            self.is_playing = True
            self.play_start_time = time.time()
            self.play_start_pos = self.position
            self.should_stop_playback = False
            
            def play():
                try:
                    # Check if we should stop before starting
                    if not self.should_stop_playback:
                        sd.play(audio_chunk, self.sr)
                        sd.wait()
                    
                    # Only set to False if this thread wasn't interrupted
                    if not self.should_stop_playback:
                        self.is_playing = False
                except Exception as e:
                    self.is_playing = False
            
            self.play_thread = threading.Thread(target=play, daemon=True)
            self.play_thread.start()

    def toggle_auto_apply(self):
        """Toggle auto-apply mode"""
        self.auto_apply = self.auto_apply_var.get()
    
    def apply_label(self):
        """Apply current label at current position"""
        current_labels = self.get_current_labels()
        if not current_labels.size:
            return
            
        label_idx = int(self.position * self.labels_per_second)
        if 0 <= label_idx < len(current_labels):
            # Apply to small window
            window = max(1, self.labels_per_second // 4)  # 0.25 second window
            start = max(0, label_idx - window//2)
            end = min(len(current_labels), label_idx + window//2)
            current_labels[start:end] = self.current_label

        self.update_copy_button_states()

    def apply_label_range(self, start_time, end_time):
        """Apply current label to a time range"""
        current_labels = self.get_current_labels()
        if not current_labels.size:
            return
            
        start_idx = int(start_time * self.labels_per_second)
        end_idx = int(end_time * self.labels_per_second)
        
        # Ensure indices are within bounds
        start_idx = max(0, start_idx)
        end_idx = min(len(current_labels), end_idx)
        
        if start_idx < end_idx:
            current_labels[start_idx:end_idx] = self.current_label

    def save_labels(self):
        """Save labels and waveform to a compressed .npz file"""
        if not self.audio_file:
            messagebox.showwarning("Warning", "No audio loaded")
            return

        try:
            # Ensure labels directory exists
            labels_dir = Path("labels")
            labels_dir.mkdir(exist_ok=True)
            
            output_path = labels_dir / Path(self.audio_file).with_suffix('.labels.npz').name
            
            # Check if file already exists
            existing_data = {}
            if output_path.exists():
                existing_data = dict(np.load(output_path))
            
            existing_data['waveform'] = self.y
            existing_data['sample_rate'] = self.sr
            existing_data['speed_labels'] = self.speed_labels
            existing_data['pattern_labels'] = self.pattern_labels
            
            # Save everything
            np.savez_compressed(output_path, **existing_data)
            
            messagebox.showinfo("Success", f"labels saved to:\n{output_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save labels:\n{str(e)}")

def main():
    root = tk.Tk()
    app = TkinterSongLabeler(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        # Stop any playing audio
        sd.stop()

if __name__ == "__main__":
    main()