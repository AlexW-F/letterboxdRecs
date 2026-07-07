<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import { fetchBackgroundCoords, type BackgroundScatter } from '$lib/api';

	let {
		height = '70vh',
		points = 4500,
		enableDrag = true,
		parallaxStrength = 1.0
	}: {
		height?: string;
		points?: number;
		enableDrag?: boolean;
		parallaxStrength?: number;
	} = $props();

	let container: HTMLDivElement;
	let loading = $state(true);
	let error = $state<string | null>(null);
	let hoveredTitle = $state<string | null>(null);
	let hoveredGenre = $state<string | null>(null);

	// Sparse labels for the few films closest to the moving camera.
	// Svelte reactivity only fires when the label MEMBERSHIP changes
	// (a film enters/leaves the candidate set). Position + opacity are
	// updated imperatively on the DOM every frame via labelEls so they
	// track the camera smoothly without per-frame Svelte diffing.
	let labels = $state<Array<{ title: string; id: string; starIdx: number }>>([]);
	type LabelState = {
		id: string;
		title: string;
		starIdx: number;
		opacity: number;       // smoothed current
		targetOpacity: number; // what we're approaching
	};
	const labelMap = new Map<string, LabelState>();
	const labelEls: Record<string, HTMLDivElement> = $state({});

	let cleanup: (() => void) | null = null;

	// Accessibility: when the user asks for reduced motion we render a single
	// static frame instead of running the drift/twinkle/pulse rAF loop.
	const reduceMotion =
		typeof window !== 'undefined' &&
		typeof window.matchMedia === 'function' &&
		window.matchMedia('(prefers-reduced-motion: reduce)').matches;

	onMount(async () => {
		let data: BackgroundScatter;
		try {
			data = await fetchBackgroundCoords(points);
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
			loading = false;
			return;
		}

		let THREE: typeof import('three');
		try {
			THREE = await import('three');
		} catch {
			error = '3D view unavailable on this device';
			loading = false;
			return;
		}

		// The galactic-web (k-NN lines) is an O(n²) build done synchronously at
		// mount — it stalls low-power phones. Skip it on small screens / sparse
		// clouds; keep it for the denser desktop scene where it reads well.
		const viewportW = typeof window !== 'undefined' ? window.innerWidth : 1280;
		const buildWeb = points >= 3000 && viewportW >= 1024;

		// ---------------------------------------------------------------
		// Scene + renderer
		// ---------------------------------------------------------------
		// Context creation throws where WebGL is disabled/unsupported — fall
		// back to the error state instead of spinning on "weaving the cloud…"
		// forever.
		let renderer: import('three').WebGLRenderer;
		try {
			renderer = new THREE.WebGLRenderer({
				alpha: true,
				antialias: true,
				powerPreference: 'high-performance'
			});
		} catch {
			error = '3D view unavailable on this device';
			loading = false;
			return;
		}
		renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
		renderer.setSize(container.clientWidth, container.clientHeight, false);
		renderer.setClearColor(0x000000, 0);
		container.appendChild(renderer.domElement);
		renderer.domElement.style.cssText =
			'width: 100%; height: 100%; display: block; touch-action: none; cursor: ' +
			(enableDrag ? 'grab' : 'default') +
			';';

		const scene = new THREE.Scene();
		scene.fog = new THREE.Fog(0x0a0c10, 18, 80);

		const camera = new THREE.PerspectiveCamera(
			72,
			container.clientWidth / container.clientHeight,
			0.1,
			200
		);
		camera.position.set(0, 0, 0);

		// ---------------------------------------------------------------
		// Center + scale the cloud
		// ---------------------------------------------------------------
		const raw = data.coords;
		let cx = 0, cy = 0, cz = 0;
		for (const [x, y, z] of raw) {
			cx += x;
			cy += y;
			cz += z;
		}
		cx /= raw.length;
		cy /= raw.length;
		cz /= raw.length;

		let maxR = 0;
		for (const [x, y, z] of raw) {
			const dx = x - cx,
				dy = y - cy,
				dz = z - cz;
			const r2 = dx * dx + dy * dy + dz * dz;
			if (r2 > maxR) maxR = r2;
		}
		maxR = Math.sqrt(maxR);
		const scale = 28 / maxR;

		const n = raw.length;
		const positions = new Float32Array(n * 3);
		const colors = new Float32Array(n * 3);
		const sizes = new Float32Array(n);
		const seeds = new Float32Array(n);

		const c = new THREE.Color();
		const popMax = Math.max(...data.popularity);
		for (let i = 0; i < n; i++) {
			positions[3 * i] = (raw[i][0] - cx) * scale;
			positions[3 * i + 1] = (raw[i][1] - cy) * scale;
			positions[3 * i + 2] = (raw[i][2] - cz) * scale;
			const hex = data.genre_colors[data.genres[i]] ?? '#cccccc';
			c.set(hex);
			const hsl = { h: 0, s: 0, l: 0 };
			c.getHSL(hsl);
			c.setHSL(hsl.h, Math.min(1, hsl.s * 1.4 + 0.1), Math.max(0.55, Math.min(0.75, hsl.l)));
			colors[3 * i] = c.r;
			colors[3 * i + 1] = c.g;
			colors[3 * i + 2] = c.b;
			const popN = Math.log1p(data.popularity[i]) / Math.log1p(popMax);
			sizes[i] = 0.55 + popN * 2.05;
			seeds[i] = Math.random();
		}

		const geometry = new THREE.BufferGeometry();
		geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
		geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
		geometry.setAttribute('aSize', new THREE.BufferAttribute(sizes, 1));
		geometry.setAttribute('aSeed', new THREE.BufferAttribute(seeds, 1));

		// ---------------------------------------------------------------
		// Star shader (galaxy core + halo)
		// ---------------------------------------------------------------
		const starMaterial = new THREE.ShaderMaterial({
			uniforms: {
				uPixelRatio: { value: Math.min(window.devicePixelRatio, 2) },
				uSize: { value: 90.0 },
				uTime: { value: 0 }
			},
			vertexShader: /* glsl */ `
				attribute float aSize;
				attribute float aSeed;
				varying vec3 vColor;
				varying float vDist;
				varying float vTwinkle;
				uniform float uPixelRatio;
				uniform float uSize;
				uniform float uTime;
				void main() {
					vColor = color;
					vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
					vDist = -mvPosition.z;
					vTwinkle = 0.85 + 0.30 * sin(uTime * 1.3 + aSeed * 18.0);
					gl_Position = projectionMatrix * mvPosition;
					gl_PointSize = uSize * aSize * uPixelRatio / max(2.0, -mvPosition.z);
				}
			`,
			fragmentShader: /* glsl */ `
				varying vec3 vColor;
				varying float vDist;
				varying float vTwinkle;
				void main() {
					vec2 d = gl_PointCoord - 0.5;
					float r = length(d) * 2.0;
					if (r > 1.0) discard;
					float halo = exp(-r * 3.4);
					float core = pow(1.0 - r, 7.0);
					vec3 col = mix(vColor, vec3(1.0), core * 0.55);
					float intensity = (halo * 0.75 + core * 2.2) * vTwinkle;
					float depthFade = 1.0 - smoothstep(22.0, 72.0, vDist);
					intensity *= 0.45 + 0.55 * depthFade;
					gl_FragColor = vec4(col * intensity, intensity);
				}
			`,
			blending: THREE.AdditiveBlending,
			depthWrite: false,
			transparent: true,
			vertexColors: true
		});
		const stars = new THREE.Points(geometry, starMaterial);
		scene.add(stars);

		// Diffuse-glow pass
		const glowMaterial = new THREE.ShaderMaterial({
			uniforms: {
				uPixelRatio: { value: Math.min(window.devicePixelRatio, 2) },
				uSize: { value: 260.0 },
				uTime: { value: 0 }
			},
			vertexShader: /* glsl */ `
				attribute float aSize;
				varying vec3 vColor;
				varying float vDist;
				uniform float uPixelRatio;
				uniform float uSize;
				void main() {
					vColor = color;
					vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
					vDist = -mvPosition.z;
					gl_Position = projectionMatrix * mvPosition;
					gl_PointSize = uSize * aSize * uPixelRatio / max(2.0, -mvPosition.z);
				}
			`,
			fragmentShader: /* glsl */ `
				varying vec3 vColor;
				varying float vDist;
				void main() {
					vec2 d = gl_PointCoord - 0.5;
					float r = length(d) * 2.0;
					if (r > 1.0) discard;
					float glow = exp(-r * 5.0) * 0.18;
					float depthFade = 1.0 - smoothstep(25.0, 80.0, vDist);
					glow *= 0.4 + 0.6 * depthFade;
					gl_FragColor = vec4(vColor * glow, glow);
				}
			`,
			blending: THREE.AdditiveBlending,
			depthWrite: false,
			transparent: true,
			vertexColors: true
		});
		const glow = new THREE.Points(geometry, glowMaterial);
		scene.add(glow);

		// ---------------------------------------------------------------
		// k-NN connection lines — the "galactic web" texture. Computed
		// once at init: each star connects to its 2 nearest neighbors if
		// they're within MAX_LINK_DIST. Filter ensures sparse regions
		// stay sparse and dense clusters web together.
		//
		// This is O(n²) and synchronous, so it's gated behind `buildWeb`
		// (desktop + dense cloud only). On phones we skip it entirely.
		// ---------------------------------------------------------------
		let links: import('three').LineSegments | null = null;
		let lineGeo: import('three').BufferGeometry | null = null;
		let lineMaterial: import('three').ShaderMaterial | null = null;

		if (buildWeb) {
			const K_LINKS = 2;
			const MAX_LINK_DIST = 1.9; // world units; cloud scaled radius ~28
			const MAX_LINK_DIST_SQ = MAX_LINK_DIST * MAX_LINK_DIST;

			const tmpPairs: Array<[number, number, number]> = []; // [i, j, distSq]
			for (let i = 0; i < n; i++) {
				let best1 = -1, best2 = -1, bd1 = Infinity, bd2 = Infinity;
				const xi = positions[3 * i],
					yi = positions[3 * i + 1],
					zi = positions[3 * i + 2];
				for (let j = 0; j < n; j++) {
					if (i === j) continue;
					const dx = xi - positions[3 * j],
						dy = yi - positions[3 * j + 1],
						dz = zi - positions[3 * j + 2];
					const d2 = dx * dx + dy * dy + dz * dz;
					if (d2 > MAX_LINK_DIST_SQ) continue;
					if (d2 < bd1) {
						bd2 = bd1;
						best2 = best1;
						bd1 = d2;
						best1 = j;
					} else if (d2 < bd2) {
						bd2 = d2;
						best2 = j;
					}
				}
				if (best1 >= 0 && i < best1) tmpPairs.push([i, best1, bd1]);
				if (K_LINKS >= 2 && best2 >= 0 && i < best2) tmpPairs.push([i, best2, bd2]);
			}

			const segs = tmpPairs.length;
			const linePositions = new Float32Array(segs * 2 * 3);
			const lineColors = new Float32Array(segs * 2 * 3);
			const lineSeeds = new Float32Array(segs * 2);
			for (let s = 0; s < segs; s++) {
				const [i, j] = tmpPairs[s];
				linePositions[s * 6 + 0] = positions[3 * i];
				linePositions[s * 6 + 1] = positions[3 * i + 1];
				linePositions[s * 6 + 2] = positions[3 * i + 2];
				linePositions[s * 6 + 3] = positions[3 * j];
				linePositions[s * 6 + 4] = positions[3 * j + 1];
				linePositions[s * 6 + 5] = positions[3 * j + 2];
				// Blend the two endpoint colors so each segment shifts naturally
				// along its length (genre A -> genre B gradients).
				lineColors[s * 6 + 0] = colors[3 * i];
				lineColors[s * 6 + 1] = colors[3 * i + 1];
				lineColors[s * 6 + 2] = colors[3 * i + 2];
				lineColors[s * 6 + 3] = colors[3 * j];
				lineColors[s * 6 + 4] = colors[3 * j + 1];
				lineColors[s * 6 + 5] = colors[3 * j + 2];
				const seed = Math.random();
				lineSeeds[s * 2] = seed;
				lineSeeds[s * 2 + 1] = seed;
			}

			lineGeo = new THREE.BufferGeometry();
			lineGeo.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
			lineGeo.setAttribute('color', new THREE.BufferAttribute(lineColors, 3));
			lineGeo.setAttribute('aSeed', new THREE.BufferAttribute(lineSeeds, 1));

			lineMaterial = new THREE.ShaderMaterial({
				uniforms: { uTime: { value: 0 } },
				vertexShader: /* glsl */ `
					attribute float aSeed;
					varying vec3 vColor;
					varying float vDist;
					varying float vPulse;
					uniform float uTime;
					void main() {
						vColor = color;
						vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
						vDist = -mvPosition.z;
						// Each link has its own pulse phase via aSeed; both endpoints
						// share the seed so the whole segment lights together.
						// pow(sin, 6) sharpens the wave into a brief flash with a
						// long dim baseline — galactic-circuit feel.
						float wave = 0.5 + 0.5 * sin(uTime * 0.85 + aSeed * 11.0);
						float flash = pow(wave, 6.0);
						vPulse = 0.18 + 3.2 * flash;
						gl_Position = projectionMatrix * mvPosition;
					}
				`,
				fragmentShader: /* glsl */ `
					varying vec3 vColor;
					varying float vDist;
					varying float vPulse;
					void main() {
						float depthFade = 1.0 - smoothstep(14.0, 55.0, vDist);
						// Hot-white peak during the flash, colored ambient otherwise.
						vec3 col = mix(vColor * 0.85, vec3(1.0), clamp(vPulse * 0.18, 0.0, 0.7));
						float a = 0.09 * depthFade * vPulse;
						gl_FragColor = vec4(col, a);
					}
				`,
				blending: THREE.AdditiveBlending,
				depthWrite: false,
				transparent: true,
				vertexColors: true
			});

			links = new THREE.LineSegments(lineGeo, lineMaterial);
			scene.add(links);
		}

		// ---------------------------------------------------------------
		// Pick
		// ---------------------------------------------------------------
		const raycaster = new THREE.Raycaster();
		raycaster.params.Points = { threshold: 0.65 };
		const pointer = new THREE.Vector2(-2, -2);

		// ---------------------------------------------------------------
		// Interaction
		// ---------------------------------------------------------------
		let isDragging = false;
		let dragX = 0, dragY = 0;
		let yaw = 0, pitch = 0;
		let parallaxX = 0, parallaxY = 0;
		let targetParallaxX = 0, targetParallaxY = 0;

		function onPointerDown(e: PointerEvent) {
			if (!enableDrag) return;
			isDragging = true;
			dragX = e.clientX;
			dragY = e.clientY;
			renderer.domElement.style.cursor = 'grabbing';
			(e.target as HTMLElement).setPointerCapture?.(e.pointerId);
		}
		function onPointerUp(e: PointerEvent) {
			if (!enableDrag) return;
			isDragging = false;
			renderer.domElement.style.cursor = 'grab';
			(e.target as HTMLElement).releasePointerCapture?.(e.pointerId);
		}
		function onPointerMove(e: PointerEvent) {
			const rect = renderer.domElement.getBoundingClientRect();
			const nx = (e.clientX - rect.left) / rect.width;
			const ny = (e.clientY - rect.top) / rect.height;
			pointer.x = nx * 2 - 1;
			pointer.y = -(ny * 2 - 1);
			// Multiplied by parallaxStrength so the landing page can dial
			// the look-follow way down for a more meditative feel.
			targetParallaxX = pointer.x * 0.25 * parallaxStrength;
			targetParallaxY = pointer.y * 0.18 * parallaxStrength;
			if (isDragging) {
				const dx = e.clientX - dragX;
				const dy = e.clientY - dragY;
				yaw -= dx * 0.005;
				pitch -= dy * 0.004;
				pitch = Math.max(-0.85, Math.min(0.85, pitch));
				dragX = e.clientX;
				dragY = e.clientY;
			}
		}
		function onPointerLeave() {
			pointer.set(-2, -2);
			targetParallaxX = 0;
			targetParallaxY = 0;
			hoveredTitle = null;
			hoveredGenre = null;
		}

		if (enableDrag) {
			renderer.domElement.addEventListener('pointerdown', onPointerDown);
			window.addEventListener('pointerup', onPointerUp);
		}
		renderer.domElement.addEventListener('pointermove', onPointerMove);
		renderer.domElement.addEventListener('pointerleave', onPointerLeave);

		// Resize
		const resize = () => {
			const w = container.clientWidth;
			const h = container.clientHeight;
			camera.aspect = w / h;
			camera.updateProjectionMatrix();
			renderer.setSize(w, h, false);
			starMaterial.uniforms.uPixelRatio.value = Math.min(window.devicePixelRatio, 2);
			glowMaterial.uniforms.uPixelRatio.value = Math.min(window.devicePixelRatio, 2);
		};
		const ro = new ResizeObserver(resize);
		ro.observe(container);

		// ---------------------------------------------------------------
		// Animation loop + nearest-label updates
		// ---------------------------------------------------------------
		const start = performance.now();
		let raf = 0;
		let lastPickTime = 0;
		let lastLabelTime = 0;
		const LABEL_K = 3; // how many labels to surface at once
		const LABEL_MAX_DIST = 12; // world units
		const LABEL_MAX_DIST_SQ = LABEL_MAX_DIST * LABEL_MAX_DIST;
		const LABEL_MIN_SCREEN_DIST = 0.18; // dedup if two labels would land within ~18% of viewport
		const projV = new THREE.Vector3();
		const lookTarget = new THREE.Vector3();

		const animate = () => {
			// Reduced-motion: freeze time at a fixed phase so the scene is
			// composed (camera off-center, some labels visible) but never moves.
			const t = reduceMotion ? 18 : (performance.now() - start) / 1000;
			starMaterial.uniforms.uTime.value = t;
			glowMaterial.uniforms.uTime.value = t;
			if (lineMaterial) lineMaterial.uniforms.uTime.value = t;

			// Slowed by 2.5–3x from before so the labels have time to be read.
			const driftR = 8;
			camera.position.x = driftR * Math.cos(t * 0.024);
			camera.position.y = driftR * 0.35 * Math.sin(t * 0.036);
			camera.position.z = driftR * Math.sin(t * 0.028);

			parallaxX += (targetParallaxX - parallaxX) * 0.05;
			parallaxY += (targetParallaxY - parallaxY) * 0.05;

			lookTarget.set(
				camera.position.x * 0.3 + (yaw + parallaxX) * 14,
				camera.position.y * 0.3 + (pitch + parallaxY) * 10,
				camera.position.z * 0.3 + 4 * Math.cos(t * 0.02)
			);
			camera.lookAt(lookTarget);

			// Cloud spin halved as well.
			stars.rotation.y = t * 0.005;
			glow.rotation.y = t * 0.005;
			if (links) links.rotation.y = t * 0.005;

			// Pick (~30 Hz)
			if (performance.now() - lastPickTime > 33 && pointer.x > -1.5) {
				lastPickTime = performance.now();
				raycaster.setFromCamera(pointer, camera);
				const hits = raycaster.intersectObject(stars, false);
				if (hits.length > 0) {
					const idx = hits[0].index ?? -1;
					if (idx >= 0 && idx < data.titles.length) {
						hoveredTitle = data.titles[idx];
						hoveredGenre = data.genres[idx];
					}
				} else {
					hoveredTitle = null;
					hoveredGenre = null;
				}
			}

			// Recompute *candidates* (which stars get a label) at ~10 Hz.
			// Positions update every frame below — this only changes the set.
			// Skipped under reduced-motion: labels need follow-up frames to
			// fade in + track, which we don't run, so we omit them entirely.
			if (!reduceMotion && performance.now() - lastLabelTime > 100) {
				lastLabelTime = performance.now();
				const camX = camera.position.x,
					camY = camera.position.y,
					camZ = camera.position.z;
				const cos = Math.cos(stars.rotation.y);
				const sin = Math.sin(stars.rotation.y);

				const found: Array<[number, number]> = []; // [idx, dist2]
				for (let i = 0; i < n; i++) {
					const lx = positions[3 * i],
						ly = positions[3 * i + 1],
						lz = positions[3 * i + 2];
					const wx = lx * cos + lz * sin;
					const wz = -lx * sin + lz * cos;
					const dx = camX - wx,
						dy = camY - ly,
						dz = camZ - wz;
					const d2 = dx * dx + dy * dy + dz * dz;
					if (d2 < LABEL_MAX_DIST_SQ) found.push([i, d2]);
				}
				found.sort((a, b) => a[1] - b[1]);

				type Candidate = { id: string; title: string; sx: number; sy: number; op: number; idx: number };
				const accepted: Candidate[] = [];
				for (const [i, d2] of found) {
					if (accepted.length >= LABEL_K) break;
					const lx = positions[3 * i],
						ly = positions[3 * i + 1],
						lz = positions[3 * i + 2];
					const wx = lx * cos + lz * sin;
					const wz = -lx * sin + lz * cos;
					projV.set(wx, ly, wz).project(camera);
					if (projV.z <= 0 || projV.z >= 1) continue;
					const sx = (projV.x + 1) * 0.5;
					const sy = (1 - projV.y) * 0.5;
					if (sx < 0.06 || sx > 0.94 || sy < 0.06 || sy > 0.94) continue;
					let tooClose = false;
					for (const a of accepted) {
						const dxs = a.sx - sx,
							dys = a.sy - sy;
						if (dxs * dxs + dys * dys < LABEL_MIN_SCREEN_DIST * LABEL_MIN_SCREEN_DIST) {
							tooClose = true;
							break;
						}
					}
					if (tooClose) continue;
					const d = Math.sqrt(d2);
					const op = Math.max(0, 1 - d / LABEL_MAX_DIST);
					accepted.push({
						id: String(i),
						title: data.titles[i],
						sx,
						sy,
						op,
						idx: i
					});
				}

				const acceptedIds = new Set(accepted.map((a) => a.id));
				let membershipChanged = false;
				for (const a of accepted) {
					const existing = labelMap.get(a.id);
					if (existing) {
						existing.targetOpacity = a.op;
					} else {
						labelMap.set(a.id, {
							id: a.id,
							title: a.title,
							starIdx: a.idx,
							opacity: 0,
							targetOpacity: a.op
						});
						membershipChanged = true;
					}
				}
				for (const [id, ls] of labelMap) {
					if (!acceptedIds.has(id)) ls.targetOpacity = 0;
				}

				if (membershipChanged) {
					labels = Array.from(labelMap.values()).map((ls) => ({
						id: ls.id,
						title: ls.title,
						starIdx: ls.starIdx
					}));
				}
			}

			// Per-frame: project each labeled star to screen and mutate its
			// DOM element imperatively. No Svelte re-render. Smooth, no lag.
			const cosY = Math.cos(stars.rotation.y);
			const sinY = Math.sin(stars.rotation.y);
			const toRemove: string[] = [];
			for (const [id, ls] of labelMap) {
				const dop = ls.targetOpacity - ls.opacity;
				if (Math.abs(dop) > 0.002) {
					ls.opacity += dop * 0.08;
				} else {
					ls.opacity = ls.targetOpacity;
				}
				if (ls.opacity < 0.01 && ls.targetOpacity === 0) {
					toRemove.push(id);
					continue;
				}
				const el = labelEls[id];
				if (!el) continue;
				const i = ls.starIdx;
				const lx = positions[3 * i],
					ly = positions[3 * i + 1],
					lz = positions[3 * i + 2];
				const wx = lx * cosY + lz * sinY;
				const wz = -lx * sinY + lz * cosY;
				projV.set(wx, ly, wz).project(camera);
				if (projV.z <= 0 || projV.z >= 1) {
					el.style.opacity = '0';
					continue;
				}
				const sx = (projV.x + 1) * 0.5 * container.clientWidth;
				const sy = (1 - projV.y) * 0.5 * container.clientHeight;
				// GPU-accelerated transform — no layout thrash, no per-frame
				// Svelte diff. Offset 18px right + center vertically.
				el.style.transform = `translate3d(${sx + 18}px, ${sy}px, 0) translateY(-50%)`;
				el.style.opacity = ls.opacity.toFixed(3);
			}
			if (toRemove.length) {
				for (const id of toRemove) labelMap.delete(id);
				labels = Array.from(labelMap.values()).map((ls) => ({
					id: ls.id,
					title: ls.title,
					starIdx: ls.starIdx
				}));
			}

			renderer.render(scene, camera);
			// Reduced-motion: render a single static frame, never reschedule.
			if (!reduceMotion) raf = requestAnimationFrame(animate);
		};
		animate();
		loading = false;

		cleanup = () => {
			cancelAnimationFrame(raf);
			ro.disconnect();
			if (enableDrag) {
				renderer.domElement.removeEventListener('pointerdown', onPointerDown);
				window.removeEventListener('pointerup', onPointerUp);
			}
			renderer.domElement.removeEventListener('pointermove', onPointerMove);
			renderer.domElement.removeEventListener('pointerleave', onPointerLeave);
			geometry.dispose();
			lineGeo?.dispose();
			starMaterial.dispose();
			glowMaterial.dispose();
			lineMaterial?.dispose();
			renderer.dispose();
			renderer.domElement.remove();
		};
	});

	onDestroy(() => {
		if (cleanup) cleanup();
	});
</script>

<div class="relative w-full" style="height: {height}; user-select: none;">
	<div bind:this={container} class="absolute inset-0"></div>

	{#if loading}
		<div
			class="absolute inset-0 grid place-items-center pointer-events-none"
			style="color: var(--ink-dim);"
		>
			<div class="text-sm flex items-center gap-2">
				<span
					class="inline-block w-1.5 h-1.5 rounded-full"
					style="background: var(--brand); box-shadow: 0 0 12px var(--brand); animation: pulse-slow 1.5s ease-in-out infinite;"
				></span>
				weaving the cloud…
			</div>
		</div>
	{/if}

	{#if error}
		<div class="absolute inset-0 grid place-items-center text-xs" style="color: #fca5a5;">
			{error}
		</div>
	{/if}

	{#if !loading && !error}
		<!-- Sparse, aesthetic floating titles near the camera. Each is a
		     tiny frosted-glass card with a leader line back to the star.
		     Positions are mutated imperatively in the rAF loop via
		     labelEls so they track the camera at 60fps with no lag. -->
		{#each labels as l (l.id)}
			<div
				bind:this={labelEls[l.id]}
				class="nearby-label"
				style="
					position: absolute;
					top: 0;
					left: 0;
					pointer-events: none;
					opacity: 0;
					will-change: transform, opacity;
				"
			>
				<span class="leader-dot"></span>
				<span class="leader-line"></span>
				<span class="label-card">
					<span class="label-text">{l.title}</span>
				</span>
			</div>
		{/each}

		<!-- Hover readout -->
		<div
			class="absolute top-3 left-1/2 -translate-x-1/2 px-3 py-1.5 rounded-full text-xs mono pointer-events-none transition"
			style="
				background: rgba(10, 12, 16, 0.78);
				border: 1px solid var(--border);
				color: {hoveredTitle ? 'var(--ink)' : 'var(--ink-faint)'};
				backdrop-filter: blur(6px);
				opacity: {hoveredTitle ? 1 : 0.6};
				white-space: nowrap;
				max-width: 80vw;
				overflow: hidden;
				text-overflow: ellipsis;
				z-index: 5;
			"
		>
			{#if hoveredTitle}
				<span>{hoveredTitle}</span>
				{#if hoveredGenre}
					<span style="color: var(--ink-faint); margin-left: 0.4rem;">· {hoveredGenre}</span>
				{/if}
			{:else}
				✦  drift through the latent space{#if enableDrag}  ✦  drag to look around{/if}
			{/if}
		</div>

		<div
			class="absolute inset-x-0 bottom-0 h-32 pointer-events-none"
			style="background: linear-gradient(180deg, transparent, var(--bg) 90%);"
		></div>
	{/if}
</div>

<style>
	.nearby-label {
		display: inline-flex;
		align-items: center;
		gap: 0;
		white-space: nowrap;
		will-change: opacity, left, top;
		/* The star anchor — leader starts here, card lives to the right. */
		margin-left: -6px;
	}

	.leader-dot {
		display: inline-block;
		width: 5px;
		height: 5px;
		border-radius: 999px;
		background: rgba(245, 247, 250, 0.85);
		box-shadow: 0 0 6px rgba(245, 247, 250, 0.55);
		flex-shrink: 0;
	}

	.leader-line {
		display: inline-block;
		width: 22px;
		height: 1px;
		background: linear-gradient(
			90deg,
			rgba(245, 247, 250, 0.55),
			rgba(245, 247, 250, 0.18)
		);
		flex-shrink: 0;
	}

	.label-card {
		display: inline-flex;
		align-items: center;
		padding: 0.32rem 0.6rem;
		border-radius: 0.55rem;
		background: rgba(10, 12, 16, 0.78);
		border: 1px solid rgba(255, 255, 255, 0.14);
		backdrop-filter: blur(10px) saturate(140%);
		-webkit-backdrop-filter: blur(10px) saturate(140%);
		box-shadow: 0 6px 20px -10px rgba(0, 0, 0, 0.7), 0 0 0 1px rgba(255, 255, 255, 0.04) inset;
		transition: transform 220ms cubic-bezier(0.22, 1, 0.36, 1);
	}

	.label-text {
		font-family: 'Playfair Display', Georgia, serif;
		font-style: italic;
		font-size: 0.84rem;
		color: rgba(245, 247, 250, 0.95);
		letter-spacing: 0.01em;
		max-width: 28ch;
		overflow: hidden;
		text-overflow: ellipsis;
		line-height: 1;
	}
</style>
