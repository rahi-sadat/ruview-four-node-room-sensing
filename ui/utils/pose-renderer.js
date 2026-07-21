// Pose Renderer Utility for WiFi-DensePose UI

export class PoseRenderer {
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.config = {
      // Rendering modes
      mode: 'skeleton', // 'skeleton', 'keypoints', 'heatmap', 'dense'
      
      // Visual settings
      showKeypoints: true,
      showSkeleton: true,
      showBoundingBox: false,
      showConfidence: true,
      showZones: true,
      showDebugInfo: false,
      
      // Colors
      skeletonColor: '#00ff00',
      keypointColor: '#ff0000',
      boundingBoxColor: '#0000ff',
      confidenceColor: '#ffffff',
      zoneColor: '#ffff00',
      
      // Sizes
      keypointRadius: 4,
      skeletonWidth: 2,
      boundingBoxWidth: 2,
      fontSize: 12,
      
      // Thresholds
      confidenceThreshold: 0.3,
      keypointConfidenceThreshold: 0.1,
      
      // Performance
      enableSmoothing: true,
      maxFps: 30,
      
      ...options
    };
    
    this.logger = this.createLogger();
    this.performanceMetrics = {
      frameCount: 0,
      lastFrameTime: 0,
      averageFps: 0,
      renderTime: 0
    };
    
    // Pose skeleton connections (COCO format, 0-indexed)
    this.skeletonConnections = [
      [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], // Head
      [5, 11], [6, 12], [5, 6], // Torso
      [5, 7], [6, 8], [7, 9], [8, 10], // Arms
      [11, 13], [12, 14], [13, 15], [14, 16] // Legs
    ];
    
    // Client-side keypoint smoothing: lerp between frames to reduce jitter.
    // Maps person index → array of {x, y} for each keypoint.
    this._smoothedKeypoints = new Map();
    this._lerpAlpha = 0.10; // 0 = frozen, 1 = instant (no smoothing)

    // Initialize rendering context
    this.initializeContext();
  }

  // Lerp a single value toward target
  _lerp(current, target, alpha) {
    return current + (target - current) * alpha;
  }

  // Get smoothed keypoint positions for a person
  _getSmoothedKeypoints(personIdx, keypoints) {
    if (!this.config.enableSmoothing || !keypoints || keypoints.length === 0) {
      return keypoints;
    }

    let prev = this._smoothedKeypoints.get(personIdx);
    if (!prev || prev.length !== keypoints.length) {
      // First frame or keypoint count changed — initialize
      prev = keypoints.map(kp => ({
        x: Number.isFinite(kp.x) ? kp.x : 0,
        y: Number.isFinite(kp.y) ? kp.y : 0,
        z: Number.isFinite(kp.z) ? kp.z : 0,
        confidence: Number.isFinite(kp.confidence) ? kp.confidence : 0,
        name: kp.name
      }));
      this._smoothedKeypoints.set(personIdx, prev);
      return keypoints;
    }

    const alpha = this._lerpAlpha;
    const smoothed = keypoints.map((kp, i) => ({
      ...kp,
      x: Number.isFinite(kp.x) ? this._lerp(prev[i].x, kp.x, alpha) : kp.x,
      y: Number.isFinite(kp.y) ? this._lerp(prev[i].y, kp.y, alpha) : kp.y,
    }));

    // Update stored positions
    this._smoothedKeypoints.set(personIdx, smoothed.map(kp => ({
      x: Number.isFinite(kp.x) ? kp.x : 0,
      y: Number.isFinite(kp.y) ? kp.y : 0,
      z: Number.isFinite(kp.z) ? kp.z : 0,
      confidence: Number.isFinite(kp.confidence) ? kp.confidence : 0,
      name: kp.name
    })));

    return smoothed;
  }

  createLogger() {
    return {
      debug: (...args) => console.debug('[RENDERER-DEBUG]', new Date().toISOString(), ...args),
      info: (...args) => console.info('[RENDERER-INFO]', new Date().toISOString(), ...args),
      warn: (...args) => console.warn('[RENDERER-WARN]', new Date().toISOString(), ...args),
      error: (...args) => console.error('[RENDERER-ERROR]', new Date().toISOString(), ...args)
    };
  }

  initializeContext() {
    this.ctx.imageSmoothingEnabled = this.config.enableSmoothing;
    this.ctx.font = `${this.config.fontSize}px Arial`;
    this.ctx.textAlign = 'left';
    this.ctx.textBaseline = 'top';
  }

  // Main render method
  render(poseData, metadata = {}) {
    const startTime = performance.now();
    
    try {
      // Clear canvas
      this.clearCanvas();
      
      console.log('🎨 [RENDERER] Rendering pose data:', poseData);
      
      const persons = Array.isArray(poseData?.persons) ? poseData.persons : [];

      if (!poseData || !Array.isArray(poseData.persons) || persons.length === 0) {
        console.log('⚠️ [RENDERER] No pose data or persons array');
        this.renderNoDataMessage(poseData);
        this.renderPoseModeBadge(poseData);
        return;
      }
      
      console.log(`👥 [RENDERER] Found ${poseData.persons.length} persons to render`);

      // Render based on mode
      switch (this.config.mode) {
        case 'skeleton':
          this.renderSkeletonMode(poseData, metadata);
          break;
        case 'keypoints':
          this.renderKeypointsMode(poseData, metadata);
          break;
        case 'heatmap':
          this.renderHeatmapMode(poseData, metadata);
          break;
        case 'dense':
          this.renderDenseMode(poseData, metadata);
          break;
        default:
          this.renderSkeletonMode(poseData, metadata);
      }

      this.renderPoseModeBadge(poseData);
      this.renderPoseDiagnostics(poseData);

      // Render debug information if enabled
      if (this.config.showDebugInfo) {
        this.renderDebugInfo(poseData, metadata);
      }

      // Update performance metrics
      this.updatePerformanceMetrics(startTime);
      
    } catch (error) {
      this.logger.error('Render error', { error: error.message });
      this.renderErrorMessage(error.message);
    }
  }

  clearCanvas() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    
    // Optional: Add background
    if (this.config.backgroundColor) {
      this.ctx.fillStyle = this.config.backgroundColor;
      this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }
  }

  getPoseMode(poseData) {
    if (poseData?.pose_mode) return poseData.pose_mode;
    if (poseData?.pose_source === 'model_inference') return 'trained';
    const persons = Array.isArray(poseData?.persons) ? poseData.persons : [];
    if (!persons.length || !persons.some(person => this.isPersonRenderable(person, 'heuristic'))) {
      return 'none';
    }
    return 'heuristic';
  }

  getPoseLabel(poseData) {
    if (poseData?.pose_label) return poseData.pose_label;
    const mode = this.getPoseMode(poseData);
    if (mode === 'trained') return 'TRAINED POSE';
    if (mode === 'heuristic') return 'HEURISTIC CSI POSE - NOT TRAINED';
    return 'NO VALID POSE';
  }

  hasDrawableCoordinates(keypoint) {
    return !!keypoint && Number.isFinite(keypoint.x) && Number.isFinite(keypoint.y);
  }

  isPersonRenderable(person, poseMode) {
    if (!person || !Array.isArray(person.keypoints)) return false;
    const hasCoordinates = person.keypoints.some(kp => this.hasDrawableCoordinates(kp));
    if (!hasCoordinates) return false;
    if (poseMode === 'heuristic') return true;
    return person.confidence === undefined || person.confidence >= this.config.confidenceThreshold;
  }

  visualKeypointConfidence(keypoint, poseMode) {
    if (!this.hasDrawableCoordinates(keypoint)) {
      return 0;
    }
    const sourceConfidence = Number.isFinite(keypoint?.confidence) ? keypoint.confidence : 0;
    if (sourceConfidence > this.config.keypointConfidenceThreshold) {
      return sourceConfidence;
    }
    return poseMode === 'heuristic' && this.hasDrawableCoordinates(keypoint) ? 0.28 : 0;
  }

  visualPersonConfidence(person, poseMode) {
    const sourceConfidence = Number.isFinite(person?.confidence) ? person.confidence : 0;
    if (sourceConfidence >= this.config.confidenceThreshold) {
      return sourceConfidence;
    }
    return poseMode === 'heuristic' && this.isPersonRenderable(person, poseMode) ? 0.42 : 0;
  }

  getPersonCenter(person) {
    const valid = (person?.keypoints || []).filter(kp => this.hasDrawableCoordinates(kp));
    if (valid.length > 0) {
      return {
        x: valid.reduce((sum, kp) => sum + kp.x, 0) / valid.length,
        y: valid.reduce((sum, kp) => sum + kp.y, 0) / valid.length
      };
    }
    if (person?.bbox) {
      return {
        x: person.bbox.x + person.bbox.width / 2,
        y: person.bbox.y + person.bbox.height / 2
      };
    }
    return { x: 320, y: 240 };
  }

  getRenderablePersons(poseData, poseMode) {
    const persons = (poseData.persons || [])
      .filter(person => this.isPersonRenderable(person, poseMode));
    if (poseMode !== 'heuristic' || persons.length <= 1) {
      return persons;
    }

    const ranked = persons
      .map((person, index) => {
        const center = this.getPersonCenter(person);
        const dx = center.x - 320;
        const dy = center.y - 260;
        const distancePenalty = Math.sqrt(dx * dx + dy * dy) / 800;
        const confidence = Number.isFinite(person.confidence) ? person.confidence : 0;
        return { person, index, score: confidence - distancePenalty };
      })
      .sort((a, b) => b.score - a.score || a.index - b.index);

    return ranked.length ? [ranked[0].person] : [];
  }

  renderPoseModeBadge(poseData) {
    const label = this.getPoseLabel(poseData);
    const mode = this.getPoseMode(poseData);
    const palette = {
      trained: { bg: 'rgba(0, 204, 136, 0.88)', fg: '#04130d' },
      heuristic: { bg: 'rgba(245, 158, 11, 0.92)', fg: '#1b1200' },
      none: { bg: 'rgba(148, 163, 184, 0.85)', fg: '#08111f' }
    };
    const colors = palette[mode] || palette.none;
    const x = 10;
    const y = 10;
    this.ctx.save();
    this.ctx.font = '12px Arial';
    const width = Math.min(this.canvas.width - 20, this.ctx.measureText(label).width + 22);
    this.ctx.fillStyle = colors.bg;
    this.ctx.fillRect(x, y, width, 24);
    this.ctx.fillStyle = colors.fg;
    this.ctx.fillText(label, x + 11, y + 6);
    this.ctx.restore();
  }

  renderPoseDiagnostics(poseData) {
    const diagnostics = poseData?.diagnostics;
    if (!diagnostics) return;
    const lines = [
      `Nodes: ${diagnostics.active_nodes ?? 0}`,
      `Keypoints: ${diagnostics.drawable_keypoints ?? 0}/${diagnostics.total_keypoints ?? 0}`,
      `Mean conf: ${((diagnostics.mean_source_confidence || 0) * 100).toFixed(1)}%`
    ];
    const x = 10;
    const y = 40;
    this.ctx.save();
    this.ctx.font = '11px Arial';
    this.ctx.fillStyle = 'rgba(0, 0, 0, 0.62)';
    this.ctx.fillRect(x, y, 150, 52);
    this.ctx.fillStyle = '#e5edf8';
    lines.forEach((line, index) => {
      this.ctx.fillText(line, x + 8, y + 7 + index * 14);
    });
    this.ctx.restore();
  }

  // Skeleton rendering mode
  renderSkeletonMode(poseData, metadata) {
    const poseMode = this.getPoseMode(poseData);
    const persons = this.getRenderablePersons(poseData, poseMode);
    
    console.log(`🦴 [RENDERER] Skeleton mode: processing ${persons.length} persons`);
    
    persons.forEach((person, index) => {
      console.log(`👤 [RENDERER] Person ${index}:`, person);
      
      if (!this.isPersonRenderable(person, poseMode)) {
        console.log(`❌ [RENDERER] Skipping person ${index} - low confidence: ${person.confidence} < ${this.config.confidenceThreshold}`);
        return; // Skip low confidence detections
      }

      // Apply client-side lerp smoothing to reduce visual jitter
      const smoothedKps = this._getSmoothedKeypoints(index, person.keypoints);

      // Render skeleton connections
      if (this.config.showSkeleton && smoothedKps) {
        this.renderSkeleton(smoothedKps, person.confidence, { poseMode });
      }

      // Render keypoints
      if (this.config.showKeypoints && smoothedKps) {
        this.renderKeypoints(smoothedKps, person.confidence, false, { poseMode });
      }

      // Render bounding box
      if (this.config.showBoundingBox && person.bbox) {
        console.log(`📦 [RENDERER] Rendering bounding box for person ${index}`);
        this.renderBoundingBox(person.bbox, person.confidence, index);
      }

      // Render confidence score
      if (this.config.showConfidence) {
        console.log(`📊 [RENDERER] Rendering confidence score for person ${index}`);
        this.renderConfidenceScore(person, index, poseMode);
      }
    });

    // Render zones if available
    if (this.config.showZones && poseData.zone_summary) {
      this.renderZones(poseData.zone_summary);
    }
  }

  // Keypoints only mode — large colored dots with labels, no skeleton lines
  renderKeypointsMode(poseData, metadata) {
    const poseMode = this.getPoseMode(poseData);
    const persons = this.getRenderablePersons(poseData, poseMode);

    persons.forEach((person, index) => {
      if (this.isPersonRenderable(person, poseMode) && person.keypoints) {
        this.renderKeypoints(person.keypoints, person.confidence, true, { poseMode });

        // Render bounding box
        if (this.config.showBoundingBox && person.bbox) {
          this.renderBoundingBox(person.bbox, person.confidence, index);
        }
        if (this.config.showConfidence) {
          this.renderConfidenceScore(person, index, poseMode);
        }
      }
    });

    if (this.config.showZones && poseData.zone_summary) {
      this.renderZones(poseData.zone_summary);
    }
  }

  // Heatmap rendering mode — Gaussian blobs around each keypoint
  renderHeatmapMode(poseData, metadata) {
    const poseMode = this.getPoseMode(poseData);
    const persons = this.getRenderablePersons(poseData, poseMode);

    persons.forEach((person, personIdx) => {
      if (!this.isPersonRenderable(person, poseMode) || !person.keypoints) return;

      const hue = (personIdx * 60) % 360; // different hue per person

      person.keypoints.forEach((kp) => {
        const visualConfidence = this.visualKeypointConfidence(kp, poseMode);
        if (visualConfidence <= 0) return;

        const cx = this.scaleX(kp.x);
        const cy = this.scaleY(kp.y);
        const radius = 30 + visualConfidence * 20;

        const grad = this.ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
        grad.addColorStop(0, `hsla(${hue}, 100%, 55%, ${visualConfidence * 0.7})`);
        grad.addColorStop(0.5, `hsla(${hue}, 100%, 45%, ${visualConfidence * 0.3})`);
        grad.addColorStop(1, `hsla(${hue}, 100%, 40%, 0)`);

        this.ctx.fillStyle = grad;
        this.ctx.fillRect(cx - radius, cy - radius, radius * 2, radius * 2);
      });

      // Light skeleton overlay so joints are connected
      if (person.keypoints) {
        this.ctx.globalAlpha = 0.25;
        this.renderSkeleton(person.keypoints, person.confidence, { poseMode });
        this.ctx.globalAlpha = 1.0;
      }

      if (this.config.showConfidence) {
        this.renderConfidenceScore(person, personIdx, poseMode);
      }
    });

    if (this.config.showZones && poseData.zone_summary) {
      this.renderZones(poseData.zone_summary);
    }
  }

  // Dense pose rendering mode — body region segmentation with filled polygons
  renderDenseMode(poseData, metadata) {
    const poseMode = this.getPoseMode(poseData);
    const persons = this.getRenderablePersons(poseData, poseMode);

    // Body part groups: [start_kp, end_kp, color]
    const bodyParts = [
      { name: 'head',      kps: [0, 1, 2, 3, 4],           color: 'rgba(255, 100, 100, 0.4)' },
      { name: 'torso',     kps: [5, 6, 12, 11],             color: 'rgba(100, 200, 255, 0.4)' },
      { name: 'left_arm',  kps: [5, 7, 9],                  color: 'rgba(100, 255, 150, 0.4)' },
      { name: 'right_arm', kps: [6, 8, 10],                 color: 'rgba(255, 200, 100, 0.4)' },
      { name: 'left_leg',  kps: [11, 13, 15],               color: 'rgba(200, 100, 255, 0.4)' },
      { name: 'right_leg', kps: [12, 14, 16],               color: 'rgba(255, 255, 100, 0.4)' },
    ];

    persons.forEach((person, personIdx) => {
      if (!this.isPersonRenderable(person, poseMode) || !person.keypoints) return;

      const kps = this._getSmoothedKeypoints(personIdx, person.keypoints);

      bodyParts.forEach((part) => {
        // Collect valid keypoints for this body part
        const points = part.kps
          .filter(i => kps[i] && this.visualKeypointConfidence(kps[i], poseMode) > 0)
          .map(i => ({ x: this.scaleX(kps[i].x), y: this.scaleY(kps[i].y) }));

        if (points.length < 2) return;

        // Draw filled region with padding around joints
        this.ctx.fillStyle = part.color;
        this.ctx.strokeStyle = part.color.replace('0.4', '0.7');
        this.ctx.lineWidth = 8;
        this.ctx.lineJoin = 'round';
        this.ctx.lineCap = 'round';

        // Draw thick path as a "region"
        this.ctx.beginPath();
        this.ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
          this.ctx.lineTo(points[i].x, points[i].y);
        }
        this.ctx.stroke();

        // Draw circles at each joint to widen the region
        points.forEach(p => {
          this.ctx.beginPath();
          this.ctx.arc(p.x, p.y, 10, 0, Math.PI * 2);
          this.ctx.fill();
        });
      });

      // Subtle keypoint dots on top
      this.renderKeypoints(kps, person.confidence, false, { poseMode });

      if (this.config.showConfidence) {
        this.renderConfidenceScore(person, personIdx, poseMode);
      }
    });

    if (this.config.showZones && poseData.zone_summary) {
      this.renderZones(poseData.zone_summary);
    }
  }

  // Render skeleton connections
  renderSkeleton(keypoints, confidence, options = {}) {
    const poseMode = options.poseMode || 'trained';
    const personConfidence = this.visualPersonConfidence({ keypoints, confidence }, poseMode);
    this.ctx.save();
    if (poseMode === 'heuristic') {
      this.ctx.setLineDash([7, 5]);
    }

    this.skeletonConnections.forEach(([pointA, pointB]) => {
      const keypointA = keypoints[pointA];
      const keypointB = keypoints[pointB];
      const confidenceA = this.visualKeypointConfidence(keypointA, poseMode);
      const confidenceB = this.visualKeypointConfidence(keypointB, poseMode);

      if (this.hasDrawableCoordinates(keypointA) && this.hasDrawableCoordinates(keypointB) &&
          confidenceA > 0 &&
          confidenceB > 0) {
        
        const x1 = this.scaleX(keypointA.x);
        const y1 = this.scaleY(keypointA.y);
        const x2 = this.scaleX(keypointB.x);
        const y2 = this.scaleY(keypointB.y);

        // Calculate line confidence based on both keypoints
        const lineConfidence = (confidenceA + confidenceB) / 2;
        
        // Variable line width based on confidence
        const lineWidth = this.config.skeletonWidth + (lineConfidence - 0.5) * 2;
        this.ctx.lineWidth = Math.max(1, Math.min(4, lineWidth));
        
        // Create gradient along the line
        const gradient = this.ctx.createLinearGradient(x1, y1, x2, y2);
        const colorA = this.addAlphaToColor(this.config.skeletonColor, confidenceA);
        const colorB = this.addAlphaToColor(this.config.skeletonColor, confidenceB);
        gradient.addColorStop(0, colorA);
        gradient.addColorStop(1, colorB);
        
        this.ctx.strokeStyle = gradient;
        this.ctx.globalAlpha = poseMode === 'heuristic'
          ? Math.max(0.42, personConfidence)
          : Math.min(personConfidence * 1.2, 1.0);
        
        // Add subtle glow for high confidence connections
        if (lineConfidence > 0.8) {
          this.ctx.shadowColor = this.config.skeletonColor;
          this.ctx.shadowBlur = 3;
        }

        this.ctx.beginPath();
        this.ctx.moveTo(x1, y1);
        this.ctx.lineTo(x2, y2);
        this.ctx.stroke();
        
        // Reset shadow
        this.ctx.shadowBlur = 0;
      }
    });

    this.ctx.restore();
    this.ctx.globalAlpha = 1.0;
  }

  // Render keypoints
  renderKeypoints(keypoints, confidence, enhancedMode = false, options = {}) {
    const poseMode = options.poseMode || 'trained';
    keypoints.forEach((keypoint, index) => {
      const visualConfidence = this.visualKeypointConfidence(keypoint, poseMode);
      if (visualConfidence > 0) {
        const x = this.scaleX(keypoint.x);
        const y = this.scaleY(keypoint.y);
        
        // Calculate radius based on confidence and keypoint importance
        const baseRadius = this.config.keypointRadius;
        const confidenceRadius = baseRadius + (visualConfidence - 0.5) * 2;
        const radius = Math.max(2, Math.min(8, confidenceRadius));
        
        // Set color based on keypoint type or confidence
        if (enhancedMode) {
          this.ctx.fillStyle = this.getKeypointColor(index, visualConfidence);
        } else {
          this.ctx.fillStyle = this.config.keypointColor;
        }
        
        // Add glow effect for high confidence keypoints
        if (visualConfidence > 0.8) {
          this.ctx.shadowColor = this.ctx.fillStyle;
          this.ctx.shadowBlur = 6;
          this.ctx.shadowOffsetX = 0;
          this.ctx.shadowOffsetY = 0;
        }
        
        this.ctx.globalAlpha = Math.min(1.0, visualConfidence + 0.3);
        
        // Draw keypoint with gradient
        const gradient = this.ctx.createRadialGradient(x, y, 0, x, y, radius);
        gradient.addColorStop(0, this.ctx.fillStyle);
        gradient.addColorStop(1, this.addAlphaToColor(this.ctx.fillStyle, 0.3));
        this.ctx.fillStyle = gradient;
        
        this.ctx.beginPath();
        this.ctx.arc(x, y, radius, 0, 2 * Math.PI);
        this.ctx.fill();
        
        // Reset shadow
        this.ctx.shadowBlur = 0;

        // Add keypoint labels in enhanced mode
        if (enhancedMode && this.config.showDebugInfo) {
          this.ctx.fillStyle = this.config.confidenceColor;
          this.ctx.font = '10px Arial';
          this.ctx.fillText(`${index}`, x + radius + 2, y - radius);
        }
      }
    });

    this.ctx.globalAlpha = 1.0;
  }

  // Render bounding box
  renderBoundingBox(bbox, confidence, personIndex) {
    const x = this.scaleX(bbox.x);
    const y = this.scaleY(bbox.y);
    const x2 = this.scaleX(bbox.x + bbox.width);
    const y2 = this.scaleY(bbox.y + bbox.height);
    const width = x2 - x;
    const height = y2 - y;

    this.ctx.strokeStyle = this.config.boundingBoxColor;
    this.ctx.lineWidth = this.config.boundingBoxWidth;
    this.ctx.globalAlpha = confidence;

    this.ctx.strokeRect(x, y, width, height);

    // Add person label
    this.ctx.fillStyle = this.config.boundingBoxColor;
    this.ctx.fillText(`Person ${personIndex + 1}`, x, y - 15);

    this.ctx.globalAlpha = 1.0;
  }

  // Render confidence score
  renderConfidenceScore(person, index, poseMode = 'trained') {
    let x, y;
    
    if (person.bbox) {
      x = this.scaleX(person.bbox.x);
      y = this.scaleY(person.bbox.y + person.bbox.height) + 5;
    } else if (person.keypoints && person.keypoints.length > 0) {
      // Use first available keypoint
      const firstKeypoint = person.keypoints.find(kp => kp.confidence > 0);
      if (firstKeypoint) {
        x = this.scaleX(firstKeypoint.x);
        y = this.scaleY(firstKeypoint.y) + 20;
      } else {
        x = 10;
        y = 30 + (index * 20);
      }
    } else {
      x = 10;
      y = 30 + (index * 20);
    }

    this.ctx.fillStyle = this.config.confidenceColor;
    const sourceConfidence = Number.isFinite(person.confidence) ? person.confidence : 0;
    const label = poseMode === 'heuristic'
      ? `Heuristic, source conf: ${(sourceConfidence * 100).toFixed(1)}%`
      : `Conf: ${(sourceConfidence * 100).toFixed(1)}%`;
    this.ctx.fillText(label, x, y);
  }

  // Render zones
  renderZones(zoneSummary) {
    Object.entries(zoneSummary).forEach(([zoneId, count], index) => {
      const y = 10 + (index * 20);
      
      this.ctx.fillStyle = this.config.zoneColor;
      this.ctx.fillText(`Zone ${zoneId}: ${count} person(s)`, 10, y);
    });
  }

  // Render debug information
  renderDebugInfo(poseData, metadata) {
    const debugInfo = [
      `Frame: ${poseData.frame_id || 'N/A'}`,
      `Timestamp: ${poseData.timestamp || 'N/A'}`,
      `Mode: ${this.getPoseLabel(poseData)}`,
      `Persons: ${poseData.persons?.length || 0}`,
      `Processing: ${poseData.processing_time_ms || 0}ms`,
      `FPS: ${this.performanceMetrics.averageFps.toFixed(1)}`,
      `Render: ${this.performanceMetrics.renderTime.toFixed(1)}ms`
    ];

    const startY = this.canvas.height - (debugInfo.length * 15) - 10;
    
    this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    this.ctx.fillRect(5, startY - 5, 200, debugInfo.length * 15 + 10);
    
    this.ctx.fillStyle = '#ffffff';
    debugInfo.forEach((info, index) => {
      this.ctx.fillText(info, 10, startY + (index * 15));
    });
  }

  // Render error message
  renderErrorMessage(message) {
    this.ctx.fillStyle = '#ff0000';
    this.ctx.font = '16px Arial';
    this.ctx.textAlign = 'center';
    this.ctx.fillText(
      `Render Error: ${message}`, 
      this.canvas.width / 2, 
      this.canvas.height / 2
    );
    this.ctx.textAlign = 'left';
    this.ctx.font = `${this.config.fontSize}px Arial`;
  }

  // Render no data message
  renderNoDataMessage(poseData = null) {
    const label = this.getPoseLabel(poseData);
    this.ctx.fillStyle = '#888888';
    this.ctx.font = '16px Arial';
    this.ctx.textAlign = 'center';
    this.ctx.fillText(
      label,
      this.canvas.width / 2, 
      this.canvas.height / 2
    );
    this.ctx.fillText(
      'Waiting for valid CSI pose coordinates',
      this.canvas.width / 2, 
      this.canvas.height / 2 + 25
    );
    this.ctx.textAlign = 'left';
    this.ctx.font = `${this.config.fontSize}px Arial`;
  }

  // Test method to verify canvas is working
  renderTestShape() {
    console.log('🔧 [RENDERER] Rendering test shape');
    this.clearCanvas();
    
    // Draw a test rectangle
    this.ctx.fillStyle = '#ff0000';
    this.ctx.fillRect(50, 50, 100, 100);
    
    // Draw a test circle
    this.ctx.fillStyle = '#00ff00';
    this.ctx.beginPath();
    this.ctx.arc(250, 100, 50, 0, 2 * Math.PI);
    this.ctx.fill();
    
    // Draw test text
    this.ctx.fillStyle = '#0000ff';
    this.ctx.font = '16px Arial';
    this.ctx.fillText('Canvas Test', 50, 200);
    
    console.log('✅ [RENDERER] Test shape rendered');
  }

  // Utility methods
  scaleX(x) {
    // If x is already in pixel coordinates (> 1), assume it's in the range 0-800
    // If x is normalized (0-1), scale to canvas width
    if (x > 1) {
      // Assume original image width of 800 pixels
      return (x / 800) * this.canvas.width;
    } else {
      return x * this.canvas.width;
    }
  }

  scaleY(y) {
    // If y is already in pixel coordinates (> 1), assume it's in the range 0-600
    // If y is normalized (0-1), scale to canvas height
    if (y > 1) {
      // Assume original image height of 600 pixels
      return (y / 600) * this.canvas.height;
    } else {
      return y * this.canvas.height;
    }
  }

  getKeypointColor(index, confidence) {
    // Color based on body part
    const colors = [
      '#ff0000', '#ff4500', '#ffa500', '#ffff00', '#adff2f', // Head/neck
      '#00ff00', '#00ff7f', '#00ffff', '#0080ff', '#0000ff', // Torso
      '#4000ff', '#8000ff', '#ff00ff', '#ff0080', '#ff0040', // Arms
      '#ff8080', '#ffb380', '#ffe680'  // Legs
    ];
    
    const color = colors[index % colors.length];
    const alpha = Math.floor(confidence * 255).toString(16).padStart(2, '0');
    return color + alpha;
  }

  addAlphaToColor(color, alpha) {
    // Convert hex color to rgba
    if (color.startsWith('#')) {
      const hex = color.slice(1);
      const r = parseInt(hex.slice(0, 2), 16);
      const g = parseInt(hex.slice(2, 4), 16);
      const b = parseInt(hex.slice(4, 6), 16);
      return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }
    // If already rgba, modify alpha
    if (color.startsWith('rgba')) {
      return color.replace(/[\d\.]+\)$/g, `${alpha})`);
    }
    // If rgb, convert to rgba
    if (color.startsWith('rgb')) {
      return color.replace('rgb', 'rgba').replace(')', `, ${alpha})`);
    }
    return color;
  }

  updatePerformanceMetrics(startTime) {
    const currentTime = performance.now();
    this.performanceMetrics.renderTime = currentTime - startTime;
    this.performanceMetrics.frameCount++;

    if (this.performanceMetrics.lastFrameTime > 0) {
      // Clamp to a minimum dt so consecutive frames within the same
      // performance.now() tick don't yield Infinity (issue #519 Bug 2).
      // 1 ms floor caps the displayed FPS at 1000 — far above any real
      // render rate, but finite so the EMA stays well-defined.
      const deltaTime = Math.max(currentTime - this.performanceMetrics.lastFrameTime, 1);
      const fps = 1000 / deltaTime;

      // Update average FPS using exponential moving average
      if (this.performanceMetrics.averageFps === 0) {
        this.performanceMetrics.averageFps = fps;
      } else {
        this.performanceMetrics.averageFps =
          (this.performanceMetrics.averageFps * 0.9) + (fps * 0.1);
      }
    }

    this.performanceMetrics.lastFrameTime = currentTime;
  }

  // Configuration methods
  updateConfig(newConfig) {
    this.config = { ...this.config, ...newConfig };
    this.initializeContext();
    this.logger.debug('Renderer configuration updated', { config: this.config });
  }

  setMode(mode) {
    this.config.mode = mode;
    this.logger.info('Render mode changed', { mode });
  }

  // Utility methods for external access
  getPerformanceMetrics() {
    return { ...this.performanceMetrics };
  }

  getConfig() {
    return { ...this.config };
  }

  // Resize handling
  resize(width, height) {
    this.canvas.width = width;
    this.canvas.height = height;
    this.initializeContext();
    this.logger.debug('Canvas resized', { width, height });
  }

  // Export frame as image
  exportFrame(format = 'png') {
    try {
      return this.canvas.toDataURL(`image/${format}`);
    } catch (error) {
      this.logger.error('Failed to export frame', { error: error.message });
      return null;
    }
  }
}

// Static utility methods
export const PoseRendererUtils = {
  // Create default configuration
  createDefaultConfig: () => ({
    mode: 'skeleton',
    showKeypoints: true,
    showSkeleton: true,
    showBoundingBox: false,
    showConfidence: true,
    showZones: true,
    showDebugInfo: false,
    skeletonColor: '#00ff00',
    keypointColor: '#ff0000',
    boundingBoxColor: '#0000ff',
    confidenceColor: '#ffffff',
    zoneColor: '#ffff00',
    keypointRadius: 4,
    skeletonWidth: 2,
    boundingBoxWidth: 2,
    fontSize: 12,
    confidenceThreshold: 0.3,
    keypointConfidenceThreshold: 0.1,
    enableSmoothing: true,
    maxFps: 30
  }),

  // Validate pose data format
  validatePoseData: (poseData) => {
    const errors = [];

    if (!poseData || typeof poseData !== 'object') {
      errors.push('Pose data must be an object');
      return { valid: false, errors };
    }

    if (!Array.isArray(poseData.persons)) {
      errors.push('Pose data must contain a persons array');
    }

    return {
      valid: errors.length === 0,
      errors
    };
  }
};
