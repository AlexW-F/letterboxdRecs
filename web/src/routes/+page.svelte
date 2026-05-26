<script lang="ts">
	import { goto } from '$app/navigation';
	import {
		Users as UsersIcon,
		ArrowRight,
		Sparkles,
		Loader2
	} from 'lucide-svelte';
	import MovieSpaceHero from '$lib/components/MovieSpaceHero.svelte';
	import { createGroup, createDemoGroup } from '$lib/api';

	let creatingGroup = $state(false);
	let creatingDemo = $state(false);
	let error = $state<string | null>(null);

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

	async function tryDemo() {
		error = null;
		creatingDemo = true;
		try {
			const g = await createDemoGroup();
			goto(`/group/recommendations?group=${g.group_id}`);
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			creatingDemo = false;
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
		points={5000}
		enableDrag={false}
		parallaxStrength={0.25}
	/>
</div>

<!-- Top scrim so the hero text reads cleanly against the galaxy -->
<div
	class="fixed inset-x-0 top-0 -z-[5] pointer-events-none"
	style="height: 60vh; background: radial-gradient(ellipse at top center, rgba(10, 12, 16, 0.55) 0%, transparent 70%);"
></div>

<section class="relative" style="z-index: 1; pointer-events: none;">
	<div
		class="anim-fade-up min-h-[78vh] flex flex-col items-center justify-center pb-12"
		style="pointer-events: none;"
	>
		<div
			class="hero-card relative px-6 sm:px-10 py-7 sm:py-9 text-center space-y-5 max-w-3xl"
		>
			<span class="chip chip-accent" style="pointer-events: auto;">
				<Sparkles size={11} />
				5,000 films · drifting through ALS latent space
			</span>
			<h1
				class="display-xl text-balance mx-auto"
				style="font-family: 'Playfair Display', Georgia, serif; font-style: italic; max-width: 24ch; line-height: 1.25;"
			>
				<span class="text-scrim">Movies</span><br />
				<span class="text-scrim"><span class="text-gradient">your whole group</span></span><br />
				<span class="text-scrim">will love.</span>
			</h1>
			<p
				class="text-balance max-w-2xl mx-auto"
				style="color: var(--ink); font-size: 1.05rem; line-height: 1.9;"
			>
				<span class="text-scrim">
					Each friend uploads their Letterboxd export. The backend folds everyone into the latent
					space you're hovering through, blends in Tag Genome relevance + TMDB plot embeddings,
					then re-ranks for diversity, popularity-debiased semantic alignment, and group
					fairness.
				</span>
			</p>
			<div
				class="flex flex-col items-center gap-3 pt-1"
				style="pointer-events: auto;"
			>
				<button
					class="btn btn-primary"
					style="font-size: 0.95rem; padding: 0.7rem 1.4rem;"
					onclick={createShareableGroup}
					disabled={creatingGroup}
				>
					{#if creatingGroup}
						<Loader2 size={16} class="animate-spin" />
						creating…
					{:else}
						<UsersIcon size={16} />
						Start a shared group
						<ArrowRight size={14} />
					{/if}
				</button>
				<button
					class="btn btn-ghost text-sm"
					style="color: var(--ink); opacity: 1;"
					onclick={() => goto('/me')}
				>
					Just me →
				</button>
				<button
					type="button"
					class="text-sm underline hover:opacity-80 transition disabled:opacity-40"
					style="color: var(--ink-muted);"
					onclick={tryDemo}
					disabled={creatingDemo}
				>
					{#if creatingDemo}
						loading sample group…
					{:else}
						or try with three sample cinephiles →
					{/if}
				</button>
				{#if error}
					<p class="text-xs" style="color: #fca5a5;">{error}</p>
				{/if}
			</div>
		</div>
	</div>

	<!-- About-the-system row underneath, still over the galaxy -->
	<div
		class="mt-2 grid grid-cols-1 md:grid-cols-3 gap-3"
		style="z-index: 1;"
	>
		{#each [
			{
				title: 'Six aggregation strategies',
				body: 'average · least_misery · most_pleasure · consensus · hybrid · and our group_taste_vector that fuses everyone into a super-user.',
				tint: 'chip-violet'
			},
			{
				title: 'Three content scorers',
				body: 'Tag Genome 2021 (1,084 curated tags) · TMDB plot sentence-embeddings · IMDb director one-hots. Per-scorer weights tunable.',
				tint: 'chip-brand'
			},
			{
				title: 'Personal 3D in /explore',
				body: 'Your folded-in taste vector projected into the same UMAP space, with your rated films visible around you.',
				tint: 'chip-accent'
			}
		] as item, i (item.title)}
			<div
				class="surface p-4 card-hover anim-fade-up"
				style="
					background: rgba(10, 12, 16, 0.65);
					backdrop-filter: blur(12px);
					-webkit-backdrop-filter: blur(12px);
					animation-delay: {120 + i * 50}ms;
					pointer-events: auto;
				"
			>
				<span class="chip {item.tint}">{item.title}</span>
				<p class="text-sm mt-3" style="color: var(--ink-muted); line-height: 1.5;">{item.body}</p>
			</div>
		{/each}
	</div>
</section>
