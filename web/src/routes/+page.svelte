<script lang="ts">
	import { goto } from '$app/navigation';
	import FileDropzone from '$lib/components/FileDropzone.svelte';
	import { uploadLetterboxd, getHealth, type Health } from '$lib/api';
	import { loadMembers, saveMembers, clearMembers, type Member } from '$lib/store';

	let members = $state<Member[]>(loadMembers());

	let name = $state('');
	let ratingsFile = $state<File | null>(null);
	let watchedFile = $state<File | null>(null);
	let uploading = $state(false);
	let error = $state<string | null>(null);
	let health = $state<Health | null>(null);

	$effect(() => {
		getHealth()
			.then((h) => (health = h))
			.catch(() => (health = null));
	});

	$effect(() => {
		saveMembers(members);
	});

	async function addMember() {
		error = null;
		if (!name.trim()) {
			error = 'Give this friend a name first.';
			return;
		}
		if (!ratingsFile) {
			error = 'Drop in a ratings.csv from your Letterboxd export.';
			return;
		}
		uploading = true;
		try {
			const upload = await uploadLetterboxd(ratingsFile, watchedFile);
			members = [...members, { name: name.trim(), hash: upload.hash, upload }];
			name = '';
			ratingsFile = null;
			watchedFile = null;
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			uploading = false;
		}
	}

	function removeMember(idx: number) {
		members = members.filter((_, i) => i !== idx);
	}

	function clearAll() {
		clearMembers();
		members = [];
	}

	function goGroupRecs() {
		goto('/group/recommendations');
	}

	function goSolo() {
		goto('/me');
	}
</script>

<section class="space-y-8">
	<div class="space-y-3">
		<h1 class="text-3xl sm:text-4xl font-semibold tracking-tight">
			Movie recommendations your whole group will love.
		</h1>
		<p class="text-white/60 max-w-2xl">
			Add a Letterboxd export for each friend. The backend folds everyone in to its trained latent
			space, picks candidates that overlap your group taste vector, then re-ranks for popularity,
			diversity, and content similarity. Six different aggregation strategies to choose from when
			you're ready to pick a film.
		</p>
		{#if health}
			<p class="text-xs text-white/30">
				API: {health.status} · catalog {health.catalog_size.toLocaleString()} films · models {health.model_name} · content {health.content_loaded ? 'on' : 'off'}
			</p>
		{:else}
			<p class="text-xs text-red-400/70">
				Can't reach the API. Run <code>docker compose up</code> or start the backend with
				<code>uvicorn src.api.main:app --port 8000</code>.
			</p>
		{/if}
	</div>

	<div class="grid grid-cols-1 md:grid-cols-3 gap-4">
		<div class="rounded-lg bg-white/5 border border-white/10 p-4 md:col-span-2">
			<h2 class="text-lg font-medium mb-3">Add a friend</h2>
			<div class="space-y-3">
				<input
					type="text"
					bind:value={name}
					placeholder="Name (e.g. alex, sara, mike)"
					class="w-full bg-black/30 border border-white/10 rounded px-3 py-2 text-sm focus:outline-none focus:border-emerald-400"
				/>
				<FileDropzone
					label="ratings.csv"
					hint="from your Letterboxd export — TMDB enrichment happens server-side (~15s first time)"
					bind:file={ratingsFile}
				/>
				<FileDropzone
					label="watched.csv (optional)"
					hint="excludes watched-but-unrated films from recs; also feeds ALS implicit signal"
					bind:file={watchedFile}
				/>
				{#if error}
					<p class="text-sm text-red-400/90">{error}</p>
				{/if}
				<button
					type="button"
					onclick={addMember}
					disabled={uploading}
					class="w-full px-4 py-2 rounded bg-emerald-500 text-emerald-950 font-medium hover:bg-emerald-400 disabled:bg-emerald-700/50 disabled:cursor-not-allowed"
				>
					{uploading ? 'uploading + enriching with TMDB…' : 'Add to group'}
				</button>
				{#if uploading}
					<p class="text-xs text-white/40">
						First upload of a new export looks up each film on TMDB (~15s for ~250 ratings).
						Re-uploads are instant — cached by content hash.
					</p>
				{/if}
			</div>
		</div>

		<div class="rounded-lg bg-white/5 border border-white/10 p-4 flex flex-col">
			<div class="flex items-center justify-between">
				<h2 class="text-lg font-medium">Group ({members.length})</h2>
				{#if members.length > 0}
					<button
						type="button"
						onclick={clearAll}
						class="text-xs text-white/40 hover:text-white/70"
					>
						clear
					</button>
				{/if}
			</div>
			<div class="mt-3 flex-1 space-y-2 overflow-auto max-h-72">
				{#if members.length === 0}
					<p class="text-sm text-white/40">No friends yet.</p>
				{/if}
				{#each members as m, i (m.hash + i)}
					<div class="flex items-start gap-2 bg-black/30 border border-white/5 rounded px-3 py-2">
						<div class="flex-1 min-w-0">
							<div class="text-sm font-medium truncate">{m.name}</div>
							<div class="text-[10px] text-white/40 truncate">
								{m.upload.n_ratings_mapped} of {m.upload.n_ratings_in} mapped · hash {m.hash.slice(0, 10)}…
							</div>
						</div>
						<button
							type="button"
							onclick={() => removeMember(i)}
							class="text-white/40 hover:text-red-400 text-sm"
							aria-label="remove"
						>
							✕
						</button>
					</div>
				{/each}
			</div>
			<div class="mt-3 space-y-2">
				<button
					type="button"
					onclick={goGroupRecs}
					disabled={members.length < 2}
					class="w-full px-4 py-2 rounded bg-sky-500 text-sky-950 font-medium hover:bg-sky-400 disabled:bg-sky-700/50 disabled:cursor-not-allowed text-sm"
				>
					Generate group recs ({members.length} members)
				</button>
				<button
					type="button"
					onclick={goSolo}
					class="w-full px-4 py-2 rounded bg-white/10 hover:bg-white/15 text-sm"
				>
					Just me →
				</button>
			</div>
		</div>
	</div>
</section>
