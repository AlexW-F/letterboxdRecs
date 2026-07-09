<script lang="ts">
	import { goto } from '$app/navigation';
	import { Users as UsersIcon, Loader2 } from 'lucide-svelte';
	import MovieSpaceHero from '$lib/components/MovieSpaceHero.svelte';
	import { createGroup } from '$lib/api';

	let creatingGroup = $state(false);
	let error = $state<string | null>(null);

	// Device-aware hero density. The hero builds an O(n²) k-NN web at mount,
	// which freezes low-power phones at ~5000 points. Scale the cloud down on
	// small / low-core / low-memory devices, up to ~3500 on capable desktops.
	// SSR-safe: defaults to a conservative count when there's no window.
	function computeHeroPoints(): number {
		if (typeof window === 'undefined') return 2000;
		const w = window.innerWidth;
		const cores = navigator.hardwareConcurrency ?? 4;
		// deviceMemory is non-standard (Chromium-only); treat absent as mid-range.
		const mem = (navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? 4;

		if (w < 640 || cores <= 4 || mem <= 4) return 2500;
		if (w < 1024 || cores <= 6 || mem <= 6) return 3000;
		return 3500;
	}

	const heroPoints = computeHeroPoints();

	const infoCards: { title: string; body: string; tint: string; href?: string }[] = [
		{
			title: 'Six ways to agree',
			body: 'Fair averages, least-misery, or fuse the whole group into one taste and let the projector decide. Pick the politics that fit your friends.',
			tint: 'chip-violet'
		},
		{
			title: 'Knows the deep cuts',
			body: '1,084 hand-curated tags plus plot embeddings under the hood — so the shortlist reaches past the same twenty films everyone recommends.',
			tint: 'chip-brand'
		},
		{
			title: 'Your taste, in 3D',
			body: 'The lights behind this page are real films. In the explore room you drift through them with your own ratings lit up around you.',
			tint: 'chip-rose',
			href: '/explore'
		}
	];

	async function createShareableGroup() {
		error = null;
		creatingGroup = true;
		try {
			const g = await createGroup();
			goto(`/group/${g.group_id}/join`);
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			creatingGroup = false;
		}
	}

</script>

<!--
  Landing page is a galaxy backdrop with content layered on top. The
  Three.js canvas is fixed full-viewport; content blocks float above
  with `pointer-events: none` on non-interactive copy so drag-to-spin
  reaches the canvas through dead space.
-->

<div
	class="fixed inset-0 -z-10"
	style="pointer-events: auto; isolation: isolate;"
>
	<MovieSpaceHero
		height="100vh"
		points={heroPoints}
		enableDrag={false}
		parallaxStrength={0.25}
	/>
</div>

<!-- Top scrim so the hero text reads cleanly against the galaxy -->
<div
	class="fixed inset-x-0 top-0 -z-[5] pointer-events-none"
	style="height: 60vh; background: radial-gradient(ellipse at top center, rgba(27, 26, 23, 0.55) 0%, transparent 70%);"
></div>

<section class="relative" style="z-index: 1; pointer-events: none;">
	<div
		class="anim-fade-up min-h-[78vh] flex flex-col items-center justify-center pb-12"
		style="pointer-events: none;"
	>
		<div
			class="hero-card relative px-6 sm:px-10 py-7 sm:py-9 text-center space-y-5 max-w-3xl"
		>
			<div
				class="board neon-green"
				style="font-size: 0.72rem; letter-spacing: 0.42em;"
			>
				— open all night —
			</div>
			<!-- No per-line scrims here: at Oswald's tight leading the cloned
			     background strips overlap the previous line's glyphs and paint
			     over them, ghosting the headline. A soft shadow reads fine. -->
			<h1 class="display-xl mx-auto" style="text-shadow: 0 2px 26px rgba(10, 9, 7, 0.9), 0 0 60px rgba(10, 9, 7, 0.7);">
				Movies your whole<br />
				group <span class="neon-amber">will love</span>
			</h1>
			<p
				class="text-balance max-w-2xl mx-auto"
				style="color: var(--ink); font-size: 1.05rem; line-height: 1.9;"
			>
				<span class="text-scrim">
					Everybody tosses in their Letterboxd, the projector does the math, and you argue the
					shortlist down to one film. Doors at eight.
				</span>
			</p>
			<div
				class="flex flex-col items-center gap-3 pt-1"
				style="pointer-events: auto;"
			>
				<button
					class="btn btn-primary"
					style="font-size: 0.9rem; padding: 0.75rem 1.6rem;"
					onclick={createShareableGroup}
					disabled={creatingGroup}
				>
					{#if creatingGroup}
						<Loader2 size={16} class="animate-spin" />
						lighting it up…
					{:else}
						<UsersIcon size={16} />
						Put tonight on the marquee
					{/if}
				</button>
				<button
					class="btn btn-ghost text-sm"
					style="color: var(--ink); opacity: 1;"
					onclick={() => goto('/me')}
				>
					just me →
				</button>
				<p class="text-xs mono" style="color: var(--ink-faint);">
					{heroPoints.toLocaleString()} films floating behind this page · every light is real
				</p>
				{#if error}
					<p class="text-xs" style="color: var(--rust);">{error}</p>
				{/if}
			</div>
		</div>
	</div>

	<!-- About-the-system row underneath, still over the galaxy -->
	<div
		class="mt-2 grid grid-cols-1 md:grid-cols-3 gap-3"
		style="z-index: 1;"
	>
		{#each infoCards as item, i (item.title)}
			<svelte:element
				this={item.href ? 'a' : 'div'}
				href={item.href}
				class="surface p-4 card-hover anim-fade-up block {i === 0 ? 'tilt-a' : i === 2 ? 'tilt-b' : ''}"
				style="
					background: rgba(27, 26, 23, 0.72);
					backdrop-filter: blur(12px);
					-webkit-backdrop-filter: blur(12px);
					animation-delay: {120 + i * 50}ms;
					pointer-events: auto;
				"
			>
				<span class="chip {item.tint}">{item.title}</span>
				<p class="text-sm mt-3" style="color: var(--ink-muted); line-height: 1.5;">{item.body}</p>
				{#if item.href}
					<p class="text-xs mt-2 board" style="color: var(--brand); letter-spacing: 0.14em;">step inside →</p>
				{/if}
			</svelte:element>
		{/each}
	</div>
</section>
