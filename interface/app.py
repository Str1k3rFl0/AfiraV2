import tkinter as tk
import json
import time
import random
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ai_model')))
from brain import AIModel

class App():
    def __init__(self, title, width, height):
        self.ai = AIModel()
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(f"{width}x{height}")
        self.root.configure(bg="#1e1e1e")
        
        self.learned_facts = self.ai.facts_learned
        
        with open('levels.json', 'r') as f:
            self.jsonLevel = json.load(f)
        
        self.header_app()
        self.chat_app()
        self.footer_app()
        
        self.add_message("Sunt Afira! AI-ul pe care tu trebuie sa il inveti de la 0\nScrie 'invata: <text> ca sa inveti modelul cu ceva!", sender="ai")
        
        self.root.mainloop()
             
    def header_app(self):
        header = tk.Frame(self.root, bg="#2c3e50", height=80)
        header.pack(fill="x")

        current_level = self.jsonLevel["levels"][0]

        left_frame = tk.Frame(header, bg="#2c3e50")
        left_frame.pack(side="left", padx=15, pady=10)

        tk.Label(
            left_frame,
            fg="white",
            bg="#2c3e50",
            font=("Arial", 12, "bold"),
            text=f"Level: {current_level['level']} - {current_level['name']} ({current_level['exp_required']} XP)"
        ).pack(anchor="w")

        center_frame = tk.Frame(header, bg="#2c3e50")
        center_frame.pack(side="left", expand=True)

        tk.Label(
            center_frame,
            text="Aceasta este Afira AI",
            fg="white",
            bg="#2c3e50",
            font=("Arial", 14, "bold")
        ).pack()

        tk.Label(
            center_frame,
            text="Invata AI-ul cum doresti",
            fg="white",
            bg="#2c3e50"
        ).pack()

        right_frame = tk.Frame(header, bg="#2c3e50")
        right_frame.pack(side="right", padx=15, pady=10)

        tk.Label(
            right_frame,
            text="Fapte invatate",
            fg="#bdc3c7",
            bg="#2c3e50",
            font=("Arial", 9)
        ).pack(anchor="e")

        self.facts_label = tk.Label(
            right_frame,
            text=str(self.learned_facts),
            fg="white",
            bg="#2c3e50",
            font=("Arial", 14, "bold")
        )
        self.facts_label.pack(anchor="e")

        canvas = tk.Canvas(self.root, height=2, highlightthickness=0, bg="#1e1e1e")
        canvas.pack(fill="x")
        
    def chat_app(self):
        self.chat_section = tk.Frame(self.root, bg="#1e1e1e")
        self.chat_section.place(x=0, y=83, width=700, height=367)
        
        self.chat_canvas = tk.Canvas(self.chat_section, bg="#1e1e1e", highlightthickness=0)
        self.chat_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.scrollbar = tk.Scrollbar(self.chat_section, orient="vertical", command=self.chat_canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.chat_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.chat_frame_inner = tk.Frame(self.chat_canvas, bg="#1e1e1e")
        self.chat_window = self.chat_canvas.create_window((0,0), window=self.chat_frame_inner, anchor="nw")
        self.chat_frame_inner.bind("<Configure>", lambda e: self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all")))
        self.chat_canvas.bind("<Configure>", lambda e: self.chat_canvas.itemconfig(self.chat_window, width=e.width))
        
    def add_message(self, text, sender="user"):
        bg_color = "#3498db" if sender == "user" else "#555"
        fg_color = "white"
        
        container = tk.Frame(self.chat_frame_inner, bg="#1e1e1e")
        container.pack(fill="x", padx=10, pady=5)
        
        label = tk.Label(
            container, 
            text=text, 
            bg=bg_color, 
            fg=fg_color, 
            wraplength=300, 
            justify="left", 
            padx=10, 
            pady=5,
            font=("Arial", 10),
            bd=0,
            relief="solid"
        )
        label.configure(highlightbackground="#000", highlightthickness=1)
        label.config(borderwidth=1, relief="ridge")
        label.pack(side="right" if sender=="user" else "left", anchor="e" if sender=="user" else "w")
        
        self.chat_canvas.update_idletasks()
        self.chat_canvas.yview_moveto(1)
        
    def footer_app(self):
        footer = tk.Frame(self.root, bg="#2c3e50")
        footer.place(x=0, y=450, width=700, height=50)
        
        user_input_text = tk.StringVar()
        
        def submit():
            text = user_input_text.get().strip()
            if text != "":
                self.add_message(text, sender="user")
                user_input_text.set("")
                
                if text.lower().startswith("invata:"):
                    fact_to_learn = text[len("invata:"):].strip()
                    response = self.ai.teach_AI(fact_to_learn)
                    self.learned_facts += 1
                    self.facts_label.config(text=str(self.learned_facts))
                    self.root.after(500, lambda: self.add_message(response, sender="ai"))
                else:
                    self.root.after(500, lambda: self.add_message("Foloseste 'invata: <text>' pentru a ma invata ceva!", sender="ai"))
            
        input_entry = tk.Entry(footer, textvariable=user_input_text, width=60, font=("Arial", 10))
        input_entry.place(x=10, y=12)
        input_entry.bind("<Return>", lambda event: submit())
        input_entry.focus_set()
        
        send_btn = tk.Button(footer, text="Send", command=submit, bg="#3498db", fg="white", font=("Arial", 10, "bold"))
        send_btn.place(x=570, y=10)
         
app = App("Afira AI", 700, 500)