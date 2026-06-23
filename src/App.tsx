// ── IMPORTS ──────────────────────────────────────────────────────
import { useEffect, useRef, useState, useCallback } from 'react'; // React hooks for state and lifecycle
import { FilesetResolver, HandLandmarker, DrawingUtils } from '@mediapipe/tasks-vision'; // MediaPipe AI for hand tracking
import { Loader2, RotateCcw, Trophy, Hand, Timer, RotateCw } from 'lucide-react'; // Icon components

// ── CONSTANTS ──────────────────────────────────────────────────
const PINCH_THRESHOLD = 0.05; // How close thumb and index must be to detect a pinch (0-1)
const FRAME_THRESHOLD = 0.1; // Minimum distance between hands to detect a frame gesture
const RESET_DWELL_MS = 1500; // How long to hold fist to reset (1.5 seconds)
const ROWS = 3; // Number of rows in the puzzle grid
const COLS = 3; // Number of columns in the puzzle grid

// ── TYPES ──────────────────────────────────────────────────────
interface Tile {
  id: number; // Unique ID for each tile (0-8)
  origX: number; // Original X position in the grid (where it should be)
  origY: number; // Original Y position in the grid (where it should be)
  currentX: number; // Current X position (where it is now)
  currentY: number; // Current Y position (where it is now)
}

type GameState = 'SCANNING' | 'PLAYING' | 'SOLVED'; // Three possible states of the game

// ── UTILITY FUNCTIONS ──────────────────────────────────────────

// Generates a shuffled puzzle state (scrambles the tiles)
function generatePuzzleState(cols: number, rows: number): Tile[] {
  const tiles: Tile[] = [];
  // Create tiles in correct order
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      tiles.push({ currentX: x, currentY: y, origX: x, origY: y, id: y * cols + x });
    }
  }
  // Fisher-Yates shuffle algorithm (randomizes tile positions)
  for (let i = tiles.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [tiles[i], tiles[j]] = [tiles[j], tiles[i]];
  }
  return tiles;
}

// Checks if all tiles are in their correct positions
function checkWinCondition(tiles: Tile[]): boolean {
  return tiles.every((tile, index) => tile.id === index);
}

// Converts milliseconds to MM:SS format
function formatTime(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const mins = Math.floor(totalSeconds / 60);
  const secs = totalSeconds % 60;
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Captures the current video frame as image data
function captureFrame(video: HTMLVideoElement, width: number, height: number): ImageData {
  const offscreen = document.createElement('canvas'); // Create temporary canvas
  offscreen.width = width;
  offscreen.height = height;
  const ctx = offscreen.getContext('2d');
  if (!ctx) throw new Error('Could not get canvas context');
  ctx.translate(width, 0); // Flip horizontally (mirror effect)
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, width, height); // Draw video frame to canvas
  return ctx.getImageData(0, 0, width, height); // Return image data
}

// Crops the captured frame to the area between the hands
function cropImage(imageData: ImageData, width: number, height: number, coords: { minX: number; maxX: number; minY: number; maxY: number }): HTMLCanvasElement {
  const { minX, maxX, minY, maxY } = coords;
  // Calculate crop coordinates (mirrored because of flip)
  const sx = (1 - maxX) * width;
  const sy = minY * height;
  const sw = ((1 - minX) * width) - sx;
  const sh = (maxY * height) - sy;

  const cropCanvas = document.createElement('canvas'); // Create canvas for cropped image
  cropCanvas.width = sw * 2; // Double size for better quality
  cropCanvas.height = sh * 2;
  const cropCtx = cropCanvas.getContext('2d');
  
  const tempCanvas = document.createElement('canvas'); // Temporary canvas for image data
  tempCanvas.width = width;
  tempCanvas.height = height;
  tempCanvas.getContext('2d')?.putImageData(imageData, 0, 0);
  
  if (cropCtx) {
    cropCtx.drawImage(tempCanvas, sx, sy, sw, sh, 0, 0, cropCanvas.width, cropCanvas.height);
  }
  
  return cropCanvas; // Return the cropped image as a canvas
}

// ── MAIN COMPONENT ─────────────────────────────────────────────
function App() {
  // ── STATE ──────────────────────────────────────────────────
  const [modelLoaded, setModelLoaded] = useState(false); // Is the AI model loaded?
  const [cameraReady, setCameraReady] = useState(false); // Is the camera running?
  const [gameState, setGameState] = useState<GameState>('SCANNING'); // Current game state
  const [error, setError] = useState<string | null>(null); // Any error messages
  const [timeElapsed, setTimeElapsed] = useState(0); // Timer in milliseconds

  // ── REFS ────────────────────────────────────────────────────
  const videoRef = useRef<HTMLVideoElement>(null); // Video element for camera feed
  const canvasRef = useRef<HTMLCanvasElement>(null); // Canvas for rendering
  const handLandmarkerRef = useRef<HandLandmarker | null>(null); // MediaPipe AI instance
  const requestRef = useRef<number | null>(null); // Animation frame ID for cleanup
  
  const puzzleTilesRef = useRef<Tile[]>([]); // Current puzzle tile arrangement
  const puzzleImageCanvasRef = useRef<HTMLCanvasElement | null>(null); // Cropped puzzle image
  const gameBoardCoordsRef = useRef<{ minX: number; maxX: number; minY: number; maxY: number } | null>(null); // Board boundaries
  
  const smoothCursorRef = useRef<{x: number, y: number}>({x: 0, y: 0}); // Smoothed cursor position
  const dragRef = useRef<{isDragging: boolean, tileIndex: number | null}>({ isDragging: false, tileIndex: null }); // Drag state
  const lastPinchTimeRef = useRef<number>(0); // When the last pinch occurred (to prevent rapid triggers)
  const lastFrameCoordsRef = useRef<{ minX: number; maxX: number; minY: number; maxY: number } | null>(null); // Last detected frame
  const fistHoldStartRef = useRef<number | null>(null); // When fist was first detected

  // ── MEDIAPIPE INITIALIZATION ──────────────────────────────
  useEffect(() => {
    const initMediaPipe = async () => {
      try {
        console.log('Loading MediaPipe...');
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
        );
        console.log('FilesetResolver loaded');
        handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`, // AI model
            delegate: "GPU" // Use GPU for faster processing
          },
          runningMode: "VIDEO", // Process video frames
          numHands: 2 // Track up to 2 hands
        });
        console.log('HandLandmarker created successfully!');
        setModelLoaded(true); // Mark AI as loaded
      } catch (err) {
        console.error('MediaPipe error:', err);
        setError("AI Model failed to load.");
      }
    };
    initMediaPipe(); // Start loading the AI
  }, []); // Empty dependency array = run once on mount

  // ── CAMERA INITIALIZATION ──────────────────────────────────
  useEffect(() => {
    const startCamera = async () => {
      if (!videoRef.current) return; // Wait for video element
      try {
        console.log('Starting camera...');
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" } // 720p, front camera
        });
        videoRef.current.srcObject = stream; // Give stream to video element
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play().then(() => {
            console.log('Camera started successfully!');
            setCameraReady(true); // Mark camera as ready
          });
        };
      } catch (err) {
        console.error('Camera error:', err);
        setError("Camera access denied.");
      }
    };
    startCamera(); // Start the camera
  }, []); // Run once on mount

  // ── TIMER ──────────────────────────────────────────────────
  useEffect(() => {
    let interval: number;
    if (gameState === 'PLAYING') { // Only run timer when playing
      const startTime = Date.now(); // When the game started
      interval = window.setInterval(() => {
        setTimeElapsed(Date.now() - startTime); // Update elapsed time
      }, 100); // Update every 100ms
    }
    return () => clearInterval(interval); // Cleanup on unmount
  }, [gameState]); // Re-run when game state changes

  // ── RESET GAME ──────────────────────────────────────────────
  const resetGame = () => {
    setGameState('SCANNING'); // Go back to scanning mode
    puzzleTilesRef.current = []; // Clear tiles
    dragRef.current = { isDragging: false, tileIndex: null }; // Reset drag
    gameBoardCoordsRef.current = null; // Clear board coords
    fistHoldStartRef.current = null; // Reset fist timer
    setTimeElapsed(0); // Reset timer
  };

  // ── RENDER PUZZLE ───────────────────────────────────────────
  const renderPuzzleGame = useCallback((
    ctx: CanvasRenderingContext2D, // Canvas context to draw on
    imageSource: ImageBitmap | HTMLCanvasElement, // The puzzle image
    tiles: Tile[], // Tile arrangement
    cols: number, // Number of columns
    rows: number, // Number of rows
    destWidth: number, // Destination width
    destHeight: number, // Destination height
    dragInfo: { index: number, x: number, y: number } | null, // Current drag info
    hoverIndex: number | null // Tile being hovered over
  ) => {
    const destTileW = destWidth / cols; // Width of each tile
    const destTileH = destHeight / rows; // Height of each tile
    const srcTileW = imageSource.width / cols; // Source tile width (from cropped image)
    const srcTileH = imageSource.height / rows; // Source tile height

    ctx.fillStyle = '#111'; // Dark background
    ctx.fillRect(0, 0, destWidth, destHeight);

    // Draw a single tile
    const drawTile = (tile: Tile, dx: number, dy: number, width: number, height: number, isDragging: boolean = false) => {
      const sx = tile.origX * srcTileW; // Source X (from original position)
      const sy = tile.origY * srcTileH; // Source Y
      ctx.save();
      if (isDragging) { // If dragging, add shadow and highlight
        ctx.shadowColor = 'rgba(0,0,0,0.5)';
        ctx.shadowBlur = 15;
        ctx.shadowOffsetY = 10;
        ctx.strokeStyle = '#ccff00'; // Neon green outline
        ctx.lineWidth = 2;
      } else {
        ctx.strokeStyle = '#ffffff'; // White outline
        ctx.lineWidth = 1;
      }
      ctx.drawImage(imageSource, sx, sy, srcTileW, srcTileH, dx, dy, width, height); // Draw the tile piece
      ctx.strokeRect(dx, dy, width, height); // Draw border
      ctx.restore();
    };

    // Draw all tiles in their current positions
    tiles.forEach((tile, currentIndex) => {
      const drawCol = currentIndex % cols; // Column in the grid
      const drawRow = Math.floor(currentIndex / cols); // Row in the grid
      const dx = drawCol * destTileW;
      const dy = drawRow * destTileH;

      if (dragInfo && dragInfo.index === currentIndex) { // If this tile is being dragged
        ctx.fillStyle = '#222'; // Dark placeholder where the tile was
        ctx.fillRect(dx, dy, destTileW, destTileH);
        ctx.strokeStyle = '#333';
        ctx.strokeRect(dx, dy, destTileW, destTileH);
      } else {
        if (dragInfo && hoverIndex === currentIndex) { // If hovering over this tile
          ctx.save();
          ctx.globalAlpha = 0.5;
          drawTile(tile, dx, dy, destTileW, destTileH); // Draw faded preview
          ctx.fillStyle = 'rgba(204, 255, 0, 0.2)'; // Highlight
          ctx.fillRect(dx, dy, destTileW, destTileH);
          ctx.strokeStyle = '#ccff00';
          ctx.lineWidth = 2;
          ctx.strokeRect(dx, dy, destTileW, destTileH);
          ctx.restore();
        } else {
          drawTile(tile, dx, dy, destTileW, destTileH); // Normal tile
        }
      }
    });

    // Draw the dragged tile (floating above the board)
    if (dragInfo) {
      const tile = tiles[dragInfo.index];
      const dragW = destTileW * 1.1; // Slightly larger
      const dragH = destTileH * 1.1;
      const dx = dragInfo.x - (dragW / 2);
      const dy = dragInfo.y - (dragH / 2);
      drawTile(tile, dx, dy, dragW, dragH, true);
    }
  }, []);

  // ── MAIN RENDER LOOP ────────────────────────────────────────
  const renderLoop = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const landmarker = handLandmarkerRef.current;

    if (!video || !canvas || !cameraReady) return; // Wait for everything to be ready

    if (video.readyState >= 2) { // Video has enough data
      // Match canvas size to video
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }
      
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      
      const width = canvas.width;
      const height = canvas.height;

      ctx.clearRect(0, 0, width, height); // Clear the canvas

      // ── RUN AI ON VIDEO ─────────────────────────────────
      let results = null;
      if (landmarker && modelLoaded) {
        results = landmarker.detectForVideo(video, performance.now()); // Detect hands in current frame
      }

      // ── SCANNING MODE ────────────────────────────────────
      if (gameState === 'SCANNING') {
        // Show the camera feed (mirrored)
        ctx.save();
        ctx.translate(width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0, width, height);
        ctx.restore();

        let validFrame = false; // Is the frame gesture detected?

        // Check if two hands are forming a frame
        if (results && results.landmarks && results.landmarks.length === 2) {
          const h1 = results.landmarks[0]; // Hand 1
          const h2 = results.landmarks[1]; // Hand 2
          const d1 = Math.hypot(h1[8].x - h1[4].x, h1[8].y - h1[4].y); // Distance between thumb and index on hand 1
          const d2 = Math.hypot(h2[8].x - h2[4].x, h2[8].y - h2[4].y); // Distance on hand 2

          // Both hands should have index and thumb far apart (forming a frame)
          if (d1 > FRAME_THRESHOLD && d2 > FRAME_THRESHOLD) {
            const allX = [h1[8].x, h1[4].x, h2[8].x, h2[4].x];
            const allY = [h1[8].y, h1[4].y, h2[8].y, h2[4].y];
            lastFrameCoordsRef.current = {
              minX: Math.min(...allX), maxX: Math.max(...allX),
              minY: Math.min(...allY), maxY: Math.max(...allY)
            };
            validFrame = true;
          }

          // Check for pinch gesture (thumb and index close together on both hands)
          if (d1 < PINCH_THRESHOLD && d2 < PINCH_THRESHOLD && lastFrameCoordsRef.current) {
            const now = Date.now();
            if (now - lastPinchTimeRef.current > 1000) { // Prevent rapid triggering
              lastPinchTimeRef.current = now;
              // Capture and crop the image between the hands
              const fullFrame = captureFrame(video, width, height);
              const cropCanvas = cropImage(fullFrame, width, height, lastFrameCoordsRef.current);
              puzzleImageCanvasRef.current = cropCanvas; // Store cropped image
              puzzleTilesRef.current = generatePuzzleState(COLS, ROWS); // Create shuffled puzzle
              gameBoardCoordsRef.current = { ...lastFrameCoordsRef.current }; // Store board position
              setGameState('PLAYING'); // Start the game!
            }
          }
        }

        // Draw the frame guide on screen
        if (lastFrameCoordsRef.current && validFrame) {
          const c = lastFrameCoordsRef.current;
          const sx = (1 - c.maxX) * width;
          const ex = (1 - c.minX) * width;
          const sy = c.minY * height;
          const ey = c.maxY * height;
          ctx.strokeStyle = '#ccff00'; // Neon green
          ctx.lineWidth = 4;
          ctx.strokeRect(sx, sy, ex-sx, ey-sy);
          ctx.fillStyle = "white";
          ctx.font = "bold 14px monospace";
          ctx.fillText("PINCH TO CAPTURE", sx, sy - 8);
        }

        // Draw hand landmarks on screen (for visual feedback)
        if (results && results.landmarks) {
          const drawingUtils = new DrawingUtils(ctx);
          for (const landmarks of results.landmarks) {
            ctx.save();
            ctx.translate(width, 0);
            ctx.scale(-1, 1);
            drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
              color: "#ffffff",
              lineWidth: 3
            });
            drawingUtils.drawLandmarks(landmarks, {
              color: "#ffffff",
              radius: 3,
              lineWidth: 1
            });
            ctx.restore();
          }
        }
      }

      // ── PLAYING / SOLVED MODE ─────────────────────────────
      else if ((gameState === 'PLAYING' || gameState === 'SOLVED') && puzzleImageCanvasRef.current && gameBoardCoordsRef.current) {
        
        ctx.fillStyle = '#111'; // Dark background
        ctx.fillRect(0, 0, width, height);

        // Calculate board position on screen
        const c = gameBoardCoordsRef.current;
        const boardSX = (1 - c.maxX) * width;
        const boardSY = c.minY * height;
        const boardW = ((1 - c.minX) * width) - boardSX;
        const boardH = (c.maxY * height) - boardSY;

        let hoverIndex = null; // Which tile is being hovered
        let isPinching = false; // Is the user pinching?
        let rawPointerX = 0; // Raw cursor X
        let rawPointerY = 0; // Raw cursor Y
        let interactingHand = null; // The hand being tracked

        // ── HAND TRACKING FOR PUZZLE ──────────────────────
        if (results && results.landmarks && results.landmarks.length > 0) {
          const hand = results.landmarks[0]; // Use first hand
          interactingHand = hand;
          const indexTip = hand[8]; // Index finger tip
          const thumbTip = hand[4]; // Thumb tip
          // Calculate cursor position (midpoint between index and thumb, mirrored)
          rawPointerX = (1 - ((indexTip.x + thumbTip.x) / 2)) * width;
          rawPointerY = ((indexTip.y + thumbTip.y) / 2) * height;
          const dist = Math.hypot(indexTip.x - thumbTip.x, indexTip.y - thumbTip.y);
          isPinching = dist < PINCH_THRESHOLD; // Pinch detected!

          // Smooth the cursor movement (reduces jitter)
          const distMove = Math.hypot(rawPointerX - smoothCursorRef.current.x, rawPointerY - smoothCursorRef.current.y);
          const alpha = distMove > 100 ? 1 : 0.4; // Responsive to quick movements
          smoothCursorRef.current.x = smoothCursorRef.current.x * (1 - alpha) + rawPointerX * alpha;
          smoothCursorRef.current.y = smoothCursorRef.current.y * (1 - alpha) + rawPointerY * alpha;
        }

        const cursorX = smoothCursorRef.current.x; // Smoothed cursor X
        const cursorY = smoothCursorRef.current.y; // Smoothed cursor Y

        // Calculate which tile the cursor is over
        const relX = cursorX - boardSX;
        const relY = cursorY - boardSY;

        if (relX >= 0 && relX <= boardW && relY >= 0 && relY <= boardH) {
          const col = Math.floor(relX / (boardW / COLS));
          const row = Math.floor(relY / (boardH / ROWS));
          if (col >= 0 && col < COLS && row >= 0 && row < ROWS) {
            hoverIndex = row * COLS + col; // Tile index under cursor
          }
        }

        // ── DRAG AND DROP LOGIC ───────────────────────────
        if (gameState === 'PLAYING') {
          if (isPinching) { // User is pinching
            if (!dragRef.current.isDragging) { // Not already dragging
              if (hoverIndex !== null) { // Hovering over a tile
                dragRef.current = { isDragging: true, tileIndex: hoverIndex }; // Start dragging
              }
            }
          } else { // User released the pinch
            if (dragRef.current.isDragging) { // Was dragging
              const startIndex = dragRef.current.tileIndex; // Tile picked up
              const endIndex = hoverIndex; // Tile dropped on
              if (startIndex !== null && endIndex !== null && startIndex !== endIndex) {
                // Swap the two tiles
                const newTiles = [...puzzleTilesRef.current];
                [newTiles[startIndex], newTiles[endIndex]] = [newTiles[endIndex], newTiles[startIndex]];
                puzzleTilesRef.current = newTiles;
                if (checkWinCondition(newTiles)) { // Check if puzzle is solved
                  setGameState('SOLVED'); // Victory!
                }
              }
              dragRef.current = { isDragging: false, tileIndex: null }; // End drag
            }
          }
        }

        // ── RENDER THE PUZZLE ──────────────────────────────
        ctx.save();
        ctx.translate(boardSX, boardSY);
        renderPuzzleGame(
          ctx,
          puzzleImageCanvasRef.current,
          puzzleTilesRef.current,
          COLS,
          ROWS,
          boardW,
          boardH,
          dragRef.current.isDragging && dragRef.current.tileIndex !== null ? {
            index: dragRef.current.tileIndex,
            x: relX,
            y: relY
          } : null, // Pass drag info
          hoverIndex // Pass hover info
        );
        ctx.strokeStyle = '#ffffff'; // Board border
        ctx.lineWidth = 4;
        ctx.strokeRect(0, 0, boardW, boardH);
        ctx.restore();

        // ── DRAW CURSOR INDICATOR ──────────────────────────
        if (results && results.landmarks && results.landmarks.length > 0) {
          ctx.beginPath();
          ctx.arc(cursorX, cursorY, 10, 0, Math.PI * 2);
          if (dragRef.current.isDragging) {
            ctx.fillStyle = '#ccff00'; // Green circle when dragging
            ctx.fill();
          } else {
            ctx.strokeStyle = '#ccff00'; // Green ring when not dragging
            ctx.lineWidth = 2;
            ctx.stroke();
          }
        }

        // ── FIST DETECTION FOR RESET ──────────────────────
        let isFist = false;
        if (interactingHand) {
          const wrist = interactingHand[0]; // Wrist landmark
          const tips = [8, 12, 16, 20]; // Finger tips
          const pips = [6, 10, 14, 18]; // Finger middle joints
          // Check if fingers are curled (tip closer to wrist than middle joint)
          const closedFingers = tips.filter((tipIdx, i) => {
            const tip = interactingHand[tipIdx];
            const pip = interactingHand[pips[i]];
            const dTip = Math.hypot(tip.x - wrist.x, tip.y - wrist.y);
            const dPip = Math.hypot(pip.x - wrist.x, pip.y - wrist.y);
            return dTip < dPip; // Tip is closer to wrist = finger is curled
          });
          isFist = closedFingers.length === 4; // All 4 fingers curled = fist
        }

        // ── RESET PROGRESS INDICATOR ──────────────────────
        if (isFist && gameState === 'PLAYING') {
          if (!fistHoldStartRef.current) {
            fistHoldStartRef.current = performance.now(); // Start timer
          }
          const elapsed = performance.now() - fistHoldStartRef.current; // Time fist has been held
          const progress = Math.min(elapsed / RESET_DWELL_MS, 1); // Progress (0-1)
          
          // Draw circular progress indicator
          const cx = width / 2;
          const cy = height / 2;
          ctx.save();
          ctx.beginPath();
          ctx.arc(cx, cy, 50, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
          ctx.fill();
          ctx.beginPath();
          ctx.arc(cx, cy, 50, -Math.PI / 2, (-Math.PI / 2) + (Math.PI * 2 * progress));
          ctx.strokeStyle = '#ccff00'; // Neon green
          ctx.lineWidth = 6;
          ctx.lineCap = 'round';
          ctx.stroke();
          ctx.fillStyle = "white";
          ctx.font = "bold 14px monospace";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText("RESETTING", cx, cy - 5);
          ctx.font = "10px monospace";
          ctx.fillText("Hold Fist", cx, cy + 10);
          ctx.restore();
          
          if (elapsed > RESET_DWELL_MS) {
            resetGame(); // Reset after 1.5 seconds
          }
        } else {
          fistHoldStartRef.current = null; // Reset timer if fist is released
        }
      }
    }

    // ── REQUEST NEXT FRAME ─────────────────────────────────
    if (requestRef.current !== null) {
      cancelAnimationFrame(requestRef.current); // Cancel old frame request
    }
    requestRef.current = requestAnimationFrame(renderLoop); // Request next frame
  }, [cameraReady, modelLoaded, gameState, renderPuzzleGame]); // Re-run when these change

  // ── START RENDER LOOP ──────────────────────────────────────
  useEffect(() => {
    requestRef.current = requestAnimationFrame(renderLoop);
    return () => { if (requestRef.current) cancelAnimationFrame(requestRef.current); };
  }, [renderLoop]);

  // ── UI RENDER ──────────────────────────────────────────────
  return (
    <div className="relative w-full h-full bg-black overflow-hidden rounded-xl">
      {/* Hidden video element for camera feed */}
      <video ref={videoRef} className="hidden" style={{ display: 'none' }} playsInline muted autoPlay />
      
      {/* Main canvas for rendering */}
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full object-cover mx-auto" />

      {/* ── BRAND / HEADER ────────────────────────────────── */}
      <div className="absolute top-4 left-1/2 -translate-x-1/2 z-40 pointer-events-none anim-slide-down">
        <div className="glass-soft rounded-full px-4 py-1.5 flex items-center gap-1.5">
          <span className="text-[10px] sm:text-xs font-bold tracking-[0.18em] uppercase text-white">AI Puzzle</span>
          <span className="text-[10px] sm:text-xs font-bold tracking-[0.18em] uppercase text-[#ccff00]">by y3shiri</span>
        </div>
      </div>

      {/* ── TIMER ───────────────────────────────────────────── */}
      {gameState === 'PLAYING' && (
        <div className="absolute top-4 left-4 z-30 flex items-center gap-2 glass-soft px-4 py-2 rounded-full anim-slide-down">
          <Timer className="w-4 h-4 text-[#ccff00]" />
          <span className="font-mono text-lg font-bold tracking-wider text-white">{formatTime(timeElapsed)}</span>
        </div>
      )}

      {/* ── INSTRUCTIONS ──────────────────────────────────── */}
      <div className="absolute top-16 right-4 z-20 flex flex-col items-end gap-2 pointer-events-none">
        <div className="text-[10px] text-white/80 glass-soft p-3 rounded-xl text-right anim-fade">
          {gameState === 'SCANNING' && (
            <>
              <p className="font-bold text-[#ccff00] mb-1">PHASE 1: CAPTURE</p>
              <p>1. Form a frame with two hands</p>
              <p>2. Pinch both hands to SNAP</p>
            </>
          )}
          {gameState === 'PLAYING' && (
            <>
              <p className="font-bold text-[#ccff00] mb-1">PHASE 2: SOLVE</p>
              <p>1. Pinch to Pick Up</p>
              <p>2. Drag & Drop to Swap</p>
              <p className="text-[#ccff00] mt-2">Hold Fist to Reset</p>
            </>
          )}
          {gameState === 'SOLVED' && (
            <p className="font-bold text-[#ccff00]">PUZZLE SOLVED!</p>
          )}
        </div>
      </div>

      {/* ── SOLVED MODAL ───────────────────────────────────── */}
      {gameState === 'SOLVED' && (
        <div className="absolute inset-0 flex flex-col items-center justify-center z-30 glass-overlay anim-fade">
          <div className="glass anim-pop rounded-3xl px-10 py-10 flex flex-col items-center w-[88%] max-w-sm">
            <div className="relative mb-5">
              <div className="absolute inset-0 blur-2xl bg-[#ccff00]/30 rounded-full" />
              <Trophy className="relative w-20 h-20 text-[#ccff00] drop-shadow-lg" />
            </div>
            <h2 className="text-3xl font-bold tracking-wide text-white mb-1">COMPLETE!</h2>
            <p className="text-zinc-400 text-[11px] uppercase tracking-[0.2em] mb-5">Puzzle solved</p>
            <div className="glass-soft rounded-2xl px-6 py-3 flex items-center gap-2 mb-8">
              <Timer className="w-5 h-5 text-[#ccff00]" />
              <span className="text-2xl font-mono font-bold text-white">{formatTime(timeElapsed)}</span>
            </div>
            <div className="flex flex-col gap-3 w-full">
              <button
                onClick={resetGame}
                className="bg-[#ccff00] hover:bg-[#b3e600] text-black font-bold py-3 px-8 rounded-full flex items-center justify-center gap-2 transition-transform hover:scale-[1.03] accent-glow pointer-events-auto cursor-pointer"
              >
                <RotateCw size={18} /> Play Again
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── RESET BUTTON ───────────────────────────────────── */}
      {gameState === 'PLAYING' && (
        <button
          onClick={resetGame}
          className="absolute bottom-6 left-6 z-20 glass-soft hover:bg-white/10 text-white p-3 rounded-full transition-colors pointer-events-auto cursor-pointer"
          title="Reset Game"
        >
          <RotateCcw size={20} />
        </button>
      )}

      {/* ── GESTURE HINT ───────────────────────────────────── */}
      {gameState === 'PLAYING' && (
        <div className="absolute bottom-6 right-6 z-10 flex items-center gap-2 text-white/80 text-xs pointer-events-none glass-soft px-3 py-1.5 rounded-full">
          <Hand className="w-4 h-4 text-[#ccff00]" />
          <span>Use Index+Thumb Pinch</span>
        </div>
      )}

      {/* ── LOADING STATES ─────────────────────────────────── */}
      {!cameraReady && !error && (
        <div className="absolute inset-0 flex flex-col items-center justify-center glass-overlay text-white z-20">
          <Loader2 className="w-10 h-10 animate-spin text-[#ccff00] mb-4" />
          <p className="text-sm tracking-wider uppercase">Initializing Camera...</p>
        </div>
      )}
      {cameraReady && !modelLoaded && !error && (
        <div className="absolute top-16 left-1/2 -translate-x-1/2 z-20 flex items-center gap-2 glass-soft px-3 py-1.5 rounded-full anim-fade">
          <Loader2 className="w-3 h-3 animate-spin text-[#ccff00]" />
          <span className="text-[10px] uppercase tracking-wide text-[#ccff00]">Loading AI...</span>
        </div>
      )}
      {error && (
        <div className="absolute inset-0 flex flex-col items-center justify-center glass-overlay text-red-400 z-30 p-4 text-center">
          <p className="font-bold">Error</p>
          <p className="text-sm">{error}</p>
        </div>
      )}
    </div>
  );
}

export default App;