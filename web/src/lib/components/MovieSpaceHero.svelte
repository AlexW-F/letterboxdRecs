<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import { fetchBackgroundCoords, type BackgroundScatter } from '$lib/api';

	let {
		height = '60vh',
		autoRotate = true,
		rotateSpeed = 0.0007,
		points = 3000
	}: {
		height?: string;
		autoRotate?: boolean;
		rotateSpeed?: number;
		points?: number;
	} = $props();

	let container: HTMLDivElement;
	let loading = $state(true);
	let error = $state<string | null>(null);
	let data = $state<BackgroundScatter | null>(null);
	let hoveredTitle = $state<string | null>(null);

	let plotlyRef: typeof import('plotly.js-dist-min') | null = null;
	let rafId: number | null = null;
	let theta = 0;
	let plotInstance: HTMLElement | null = null;

	function startRotate() {
		if (!autoRotate || !plotInstance || !plotlyRef) return;
		const radius = 1.85;
		const tick = () => {
			theta += rotateSpeed * 1000 / 60;
			const eye = {
				x: radius * Math.cos(theta),
				y: radius * Math.sin(theta),
				z: 0.8
			};
			plotlyRef!.relayout(plotInstance!, { 'scene.camera.eye': eye } as never);
			rafId = requestAnimationFrame(tick);
		};
		rafId = requestAnimationFrame(tick);
	}

	function stopRotate() {
		if (rafId !== null) cancelAnimationFrame(rafId);
		rafId = null;
	}

	onMount(async () => {
		try {
			data = await fetchBackgroundCoords(points);
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
			loading = false;
			return;
		}
		// Lazy-load plotly only when this component mounts so initial route
		// transitions stay fast for users who never see it.
		const Plotly = await import('plotly.js-dist-min');
		plotlyRef = Plotly;

		const traces: Record<string, { x: number[]; y: number[]; z: number[]; titles: string[] }> = {};
		for (let i = 0; i < data!.n; i++) {
			const genre = data!.genres[i];
			if (!traces[genre]) traces[genre] = { x: [], y: [], z: [], titles: [] };
			traces[genre].x.push(data!.coords[i][0]);
			traces[genre].y.push(data!.coords[i][1]);
			traces[genre].z.push(data!.coords[i][2]);
			traces[genre].titles.push(data!.titles[i]);
		}

		const plotData = Object.entries(traces).map(([genre, t]) => ({
			type: 'scatter3d' as const,
			mode: 'markers' as const,
			x: t.x,
			y: t.y,
			z: t.z,
			text: t.titles,
			hovertemplate: '%{text}<extra></extra>',
			name: genre,
			marker: {
				size: 3,
				color: data!.genre_colors[genre] ?? '#cccccc',
				opacity: 0.78,
				line: { width: 0 }
			}
		}));

		const layout = {
			showlegend: false,
			paper_bgcolor: 'rgba(0,0,0,0)',
			plot_bgcolor: 'rgba(0,0,0,0)',
			margin: { l: 0, r: 0, t: 0, b: 0 },
			scene: {
				bgcolor: 'rgba(0,0,0,0)',
				xaxis: {
					showbackground: false,
					showgrid: false,
					zeroline: false,
					showticklabels: false,
					title: { text: '' }
				},
				yaxis: {
					showbackground: false,
					showgrid: false,
					zeroline: false,
					showticklabels: false,
					title: { text: '' }
				},
				zaxis: {
					showbackground: false,
					showgrid: false,
					zeroline: false,
					showticklabels: false,
					title: { text: '' }
				},
				camera: { eye: { x: 1.85, y: 0, z: 0.8 } },
				aspectmode: 'cube'
			},
			hoverlabel: {
				bgcolor: 'rgba(10,12,16,0.92)',
				bordercolor: 'rgba(167,139,250,0.55)',
				font: { color: '#f5f7fa', size: 12, family: 'Inter, sans-serif' }
			}
		};

		const config = {
			displayModeBar: false,
			responsive: true,
			scrollZoom: false,
			displaylogo: false
		};

		await Plotly.newPlot(container, plotData as never, layout as never, config);
		plotInstance = container;
		loading = false;

		(plotInstance as unknown as { on: (ev: string, h: (d: unknown) => void) => void }).on?.(
			'plotly_hover',
			(ev: unknown) => {
				const e = ev as { points?: Array<{ text: string }> };
				hoveredTitle = e.points?.[0]?.text ?? null;
				stopRotate();
			}
		);
		(plotInstance as unknown as { on: (ev: string, h: () => void) => void }).on?.(
			'plotly_unhover',
			() => {
				hoveredTitle = null;
				if (autoRotate) startRotate();
			}
		);
		(plotInstance as unknown as { on: (ev: string, h: () => void) => void }).on?.(
			'plotly_relayouting',
			() => {
				stopRotate();
			}
		);

		if (autoRotate) startRotate();
	});

	onDestroy(() => {
		stopRotate();
		if (plotInstance && plotlyRef) {
			plotlyRef.purge(plotInstance);
		}
	});
</script>

<div class="relative w-full" style="height: {height};">
	<div bind:this={container} class="absolute inset-0"></div>

	{#if loading}
		<div
			class="absolute inset-0 grid place-items-center pointer-events-none"
			style="color: var(--ink-dim);"
		>
			<div class="text-sm flex items-center gap-2">
				<span class="inline-block w-1.5 h-1.5 rounded-full" style="background: var(--brand); box-shadow: 0 0 12px var(--brand); animation: pulse-slow 1.5s ease-in-out infinite;"></span>
				loading {points.toLocaleString()} films…
			</div>
		</div>
	{/if}

	{#if error}
		<div class="absolute inset-0 grid place-items-center text-xs" style="color: #fca5a5;">
			{error}
		</div>
	{/if}

	{#if !loading && !error}
		<!-- Hover readout — minimal, floating, centered top -->
		<div
			class="absolute top-3 left-1/2 -translate-x-1/2 px-3 py-1.5 rounded-full text-xs mono pointer-events-none transition"
			style="background: rgba(10,12,16,0.78); border: 1px solid var(--border); color: {hoveredTitle ? 'var(--ink)' : 'var(--ink-faint)'}; backdrop-filter: blur(6px); opacity: {hoveredTitle ? 1 : 0.55};"
		>
			{hoveredTitle ?? '✦  hover any star  ✦  drag to spin'}
		</div>

		<!-- Edge fades so the cloud blends into the page -->
		<div
			class="absolute inset-0 pointer-events-none"
			style="background: radial-gradient(ellipse at center, transparent 50%, var(--bg) 92%);"
		></div>
	{/if}
</div>
