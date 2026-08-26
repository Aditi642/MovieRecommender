# 🎬 CineMatch 

A sleek, user-friendly movie recommendation application designed to help users discover their next favorite film or TV show based on genres, moods, and real-time streaming availability.

## 🚀 Features
*   **Personalized Recommendations**: Tailored suggestions driven by your preferred genres, moods, and viewing history.
*   **Interactive Camera Roll Gallery**: A visual photography-inspired camera roll component (`🎞️`) to scan through movie posters and media assets.
*   **Platform Filtering**: Filter search results instantly by your active streaming services (Netflix, Hulu, Prime Video, etc.).
*   **Modern Cinematic UI**: Beautiful, lightweight interface featuring an updated inline-styled brand identity.

---

## 🚀 Live Demo

### Frontend
🔗 **(https://movie-recommender-theta-three.vercel.app/)**

---


## 🏗️ System Architecture

CineMatch is built using a modern decoupled tier-based architecture to ensure fast content rendering and highly responsive user interactions.

```
                  ┌────────────────────────┐
                  │       Client UI        │
                  │  (HTML5 / CSS3 / JS)   │
                  └───────────┬────────────┘
                              │
                    HTTPS REST Requests
                              │
                              ▼
                  ┌────────────────────────┐
                  │   API Gateway / Edge   │
                  │   (Routing & Auth)     │
                  └───────────┬────────────┘
                              │
             ┌────────────────┴────────────────┐
             ▼                                 ▼
┌────────────────────────┐        ┌────────────────────────┐
│ Recommendation Engine  │        │  Media Discovery Service│
│ (Filtering & Profiles) │        │ (Poster Assets Gallery)│
└────────────┬───────────┘        └────────────┬───────────┘
             │                                 │
             └────────────────┬────────────────┘
                              ▼
                  ┌────────────────────────┐
                  │      Data Layer        │
                  │  (Caching & Metadata)  │
                  └────────────────────────┘
```

### Architectural Component Breakdown

1.  **Presentation Layer (Frontend)**:
    *   Single Page Application (SPA) skeleton leveraging optimized styling primitives.
    *   Interactive UI features including custom CSS flex-aligned layouts (`CineMatch` branding and inline asset components).

2.  **Application Layer (Backend Microservices)**:
    *   **Recommendation Engine**: Core logic validating criteria matrix strings (e.g., matching "Feel-good" + "Sci-fi").
    *   **Media Discovery Service**: Controls metadata extraction, transforming database text records into a chronological visual camera roll.

3.  **Data & Integration Layer**:
    *   Abstracted content layer parsing external streaming indexes and media assets.

---

## 🛠️ Getting Started

### Prerequisites
*   A modern web browser (Chrome, Safari, Edge, Firefox).
*   A local text editor (VS Code, Sublime Text).

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/cinematch.git
   ```
2. Navigate to the project folder:
   ```bash
   cd cinematch
   ```
3. Open `index.html` in your favorite web browser to view the application.

---


