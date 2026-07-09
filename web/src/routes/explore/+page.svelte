<script lang="ts">
	import { Sparkles, Loader2 } from 'lucide-svelte';
	import Avatar from '$lib/components/Avatar.svelte';
	import { fetchPersonalizedVizHTML } from '$lib/api';
	import { loadMembers, type Member } from '$lib/store';

	const members = $state<Member[]>(loadMembers());

	let selectedHash = $state<string | null>(members[0]?.hash ?? null);
	let selectedLabel = $derived(
		members.find((m) => m.hash === selectedHash)?.name ?? 'you'
	);

	let personalizedHTML = $state<string | null>(null);
	let loading = $state(false);
	let error = $state<string | null>(null);

	async function loadFor(hash: string, label: string) {
		loading = true;
		error = null;
		personalizedHTML = null;
		try {
			personalizedHTML = await fetchPersonalizedVizHTML(hash, label);
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			loading = false;
		}
	}

	$effect(() => {
		if (selectedHash) loadFor(selectedHash, selectedLabel);
	});
</script>

<svelte:head>
	<title>Your taste in 3D · movienight</title>
</svelte:head>

<section class="space-y-4 anim-fade-up">
	<header>
		<span class="chip chip-accent">
			<Sparkles size={11} />
			3D · UMAP projection of ALS latent space
		</span>
		<h1 class="display-md mt-2">
			Your taste in <span class="text-gradient">3D</span>.
		</h1>
		<p class="text-sm max-w-3xl" style="color: var(--ink-muted);">
			Each dot is a film, positioned by ALS latent factors projected to 3D via UMAP. Color is
			primary genre. Your rated films are bigger, colored by rating; your folded-in taste vector
			is the black diamond. Drag to rotate, scroll to zoom, hover for titles.
		</p>
	</header>

	{#if members.length > 0}
		<div class="surface p-3 flex items-center gap-3 flex-wrap">
			<span class="text-[10px] uppercase tracking-[0.14em]" style="color: var(--ink-faint);">
				view
			</span>
			<div class="flex gap-1.5 flex-wrap">
				{#each members as m (m.hash)}
					<button
						type="button"
						onclick={() => (selectedHash = m.hash)}
						class="flex items-center gap-1.5 pl-1 pr-3 py-1 rounded-full text-sm transition"
						style={selectedHash === m.hash
							? 'background: var(--brand-dim); border: 1px solid rgba(201, 165, 84, 0.6); color: var(--brand);'
							: 'background: var(--surface); border: 1px solid var(--border); color: var(--ink-muted);'}
					>
						<Avatar name={m.name} size={22} />
						{m.name}
					</button>
				{/each}
			</div>
			<span class="text-xs ml-auto" style="color: var(--ink-faint);">
				{members.length} uploaded
			</span>
		</div>
	{:else}
		<div
			class="surface p-4 text-sm flex items-center gap-3"
			style="background: var(--accent-dim); border-color: rgba(201, 165, 84, 0.4);"
		>
			<Sparkles size={16} style="color: var(--brand);" />
			<div>
				No uploads yet — showing the demo projection.
				<a href="/" class="underline">Add a friend</a>
				to see your own.
			</div>
		</div>
	{/if}

	<div
		class="surface relative overflow-hidden"
		style="border-radius: 1rem;"
	>
		{#if loading}
			<div
				class="absolute inset-0 grid place-items-center z-10 pointer-events-none"
				style="background: rgba(27, 26, 23, 0.55); backdrop-filter: blur(4px);"
			>
				<div class="flex flex-col items-center gap-2 text-sm" style="color: var(--ink-muted);">
					<div class="flex items-center gap-3">
						<Loader2 size={18} class="animate-spin" style="color: var(--violet);" />
						projecting <em>{selectedLabel}</em> into the latent space…
					</div>
					<p class="text-xs" style="color: var(--ink-faint);">
						First projection can take up to a minute on a cold server — cached after that.
					</p>
				</div>
			</div>
		{/if}
		{#if error}
			<div class="p-4 text-sm" style="color: var(--rust);">
				Couldn't project <em>{selectedLabel}</em>: {error}
				<span style="color: var(--ink-faint);">— showing the demo projection instead.</span>
			</div>
		{/if}
		{#if personalizedHTML && !error}
			<iframe
				srcdoc={personalizedHTML}
				title="3D movie space — {selectedLabel}"
				class="w-full block"
				style="height: 80vh; border: 0; background: #1b1a17;"
			></iframe>
		{:else if !loading}
			<iframe
				src="/movie_space.html"
				title="3D movie space (demo)"
				class="w-full block"
				style="height: 80vh; border: 0; background: #1b1a17;"
			></iframe>
		{/if}
	</div>

	<p class="text-xs" style="color: var(--ink-faint);">
		Background is 8,000 films with ≥100 ratings. Cached server-side per upload — the first
		projection can take up to a minute on a cold server; repeat loads are instant.
	</p>
</section>
