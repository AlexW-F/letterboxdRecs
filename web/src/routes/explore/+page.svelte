<script lang="ts">
	import { fetchPersonalizedVizHTML } from '$lib/api';
	import { loadMembers, type Member } from '$lib/store';

	const members = $state<Member[]>(loadMembers());

	// Currently-selected member to render. Defaults to first uploaded
	// member; falls back to the static alex demo if no uploads exist.
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

<section class="space-y-4">
	<header>
		<h1 class="text-2xl font-semibold tracking-tight">Movie space</h1>
		<p class="text-sm text-white/50">
			Each dot is a film, positioned by its ALS latent factors projected to 3D via UMAP. Color
			encodes primary genre. Your rated films are bigger, colored by your rating; your folded-in
			taste vector is the black diamond.
		</p>
	</header>

	{#if members.length > 0}
		<div class="rounded-lg bg-white/5 border border-white/10 p-3 flex items-center gap-3 flex-wrap">
			<span class="text-xs uppercase tracking-wider text-white/40">view</span>
			<div class="flex gap-2 flex-wrap">
				{#each members as m (m.hash)}
					<button
						type="button"
						onclick={() => (selectedHash = m.hash)}
						class="px-3 py-1.5 rounded text-sm border transition
							{selectedHash === m.hash
							? 'bg-emerald-500/20 border-emerald-400/70 text-emerald-100'
							: 'border-white/15 text-white/70 hover:border-white/30'}"
					>
						{m.name}
					</button>
				{/each}
			</div>
			<span class="text-xs text-white/40 ml-auto">
				{members.length} uploaded — pick one to see their taste in the space
			</span>
		</div>
	{:else}
		<div class="rounded-lg bg-amber-500/10 border border-amber-400/30 p-3 text-sm">
			No uploads yet — showing the demo (alex's) projection. <a href="/" class="underline">Add a friend</a>
			to see your own.
		</div>
	{/if}

	<div class="rounded-lg border border-white/10 bg-black/30 overflow-hidden relative">
		{#if loading}
			<div class="absolute inset-0 grid place-items-center bg-black/50 z-10 pointer-events-none">
				<div class="text-sm text-white/70">projecting <em>{selectedLabel}</em> into the latent space…</div>
			</div>
		{/if}
		{#if error}
			<div class="p-4 text-red-400 text-sm">{error}</div>
		{/if}
		{#if personalizedHTML && !error}
			<iframe
				srcdoc={personalizedHTML}
				title="3D movie space — {selectedLabel}"
				class="w-full"
				style="height: 80vh; border: 0; background: #0b0d11;"
			></iframe>
		{:else if !loading && !error}
			<!-- Fallback static demo (alex) -->
			<iframe
				src="/movie_space.html"
				title="3D movie space (demo)"
				class="w-full"
				style="height: 80vh; border: 0; background: #0b0d11;"
			></iframe>
		{/if}
	</div>

	<p class="text-xs text-white/40">
		Background is a sample of 8,000 popular-enough films (≥100 ratings). Drag to rotate, scroll to
		zoom, hover for titles. Cached server-side per upload — first load takes ~1s, subsequent loads
		are instant.
	</p>
</section>
