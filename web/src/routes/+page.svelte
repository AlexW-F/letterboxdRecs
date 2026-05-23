<script lang="ts">
	import { goto } from '$app/navigation';
	import { flip } from 'svelte/animate';
	import { fly, slide } from 'svelte/transition';
	import {
		UserPlus,
		Users as UsersIcon,
		ArrowRight,
		Trash2,
		Sparkles,
		Loader2,
		FileText
	} from 'lucide-svelte';
	import FileDropzone from '$lib/components/FileDropzone.svelte';
	import Avatar from '$lib/components/Avatar.svelte';
	import MovieSpaceHero from '$lib/components/MovieSpaceHero.svelte';
	import { uploadLetterboxd } from '$lib/api';
	import { loadMembers, saveMembers, clearMembers, type Member } from '$lib/store';

	let members = $state<Member[]>(loadMembers());

	let name = $state('');
	let ratingsFile = $state<File | null>(null);
	let watchedFile = $state<File | null>(null);
	let uploading = $state(false);
	let error = $state<string | null>(null);

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
</script>

<section class="relative">
	<!-- 3D hero — fly through the actual ALS latent space rendered with
	     Three.js. Drift camera + mouse parallax + drag-to-look + hover
	     picking via raycaster. Soft additive-blended stars with depth fog. -->
	<div class="relative -mx-4 sm:-mx-6 mb-2 anim-fade-up">
		<MovieSpaceHero height="68vh" points={4500} />
	</div>

	<div class="absolute inset-0 -z-10 overflow-hidden pointer-events-none">
		<div
			class="absolute -top-24 left-1/3 w-[60vw] h-[60vw] max-w-[700px] max-h-[700px] rounded-full"
			style="background: radial-gradient(circle, rgba(52, 211, 153, 0.12), transparent 60%); filter: blur(40px);"
		></div>
		<div
			class="absolute top-32 right-0 w-[40vw] h-[40vw] max-w-[500px] max-h-[500px] rounded-full"
			style="background: radial-gradient(circle, rgba(167, 139, 250, 0.10), transparent 60%); filter: blur(50px);"
		></div>
	</div>

	<div class="anim-fade-up space-y-4 pb-10" style="animation-delay: 80ms;">
		<span class="chip chip-accent">
			<Sparkles size={11} />
			4,500 films above · drift, drag, hover · the latent space your taste lives in
		</span>
		<h1 class="display-xl text-balance" style="font-family: 'Instrument Serif', Georgia, serif; font-style: italic;">
			Movie recommendations <br class="hidden md:block" />
			<span class="text-gradient">your whole group</span> will love.
		</h1>
		<p class="text-balance max-w-2xl" style="color: var(--ink-muted); font-size: 1.05rem; line-height: 1.55;">
			Each friend uploads their Letterboxd export. The backend folds everyone into the 64-dim
			latent space you see above, blends in Tag Genome 2021 relevance and TMDB plot embeddings,
			then re-ranks for diversity and popularity-debiased semantic alignment. Six aggregation
			strategies including our <span class="chip chip-violet ml-1">group_taste_vector</span>
			which picks films <em>nobody would have chosen alone</em> but everyone will love.
		</p>
	</div>

	<div class="grid grid-cols-1 md:grid-cols-5 gap-4 anim-fade-up" style="animation-delay: 80ms;">
		<div class="surface p-5 md:col-span-3 space-y-4">
			<div class="flex items-center gap-2">
				<UserPlus size={18} style="color: var(--brand);" />
				<h2 class="text-lg font-medium">Add a friend</h2>
			</div>
			<input
				type="text"
				bind:value={name}
				placeholder="Their name (e.g. alex, sara, mike)"
				class="input"
				disabled={uploading}
			/>
			<FileDropzone
				label="ratings.csv"
				hint="from your Letterboxd export · TMDB enrichment happens server-side"
				bind:file={ratingsFile}
			/>
			<FileDropzone
				label="watched.csv (optional)"
				hint="excludes already-watched films + feeds ALS implicit signal"
				bind:file={watchedFile}
			/>
			{#if error}
				<p
					class="text-sm rounded-md px-3 py-2"
					style="background: var(--rose-dim); border: 1px solid rgba(248, 113, 113, 0.3); color: #fecaca;"
				>
					{error}
				</p>
			{/if}
			<button class="btn btn-primary w-full" onclick={addMember} disabled={uploading}>
				{#if uploading}
					<Loader2 size={16} class="animate-spin" />
					Uploading + enriching with TMDB…
				{:else}
					<UserPlus size={16} />
					Add to group
				{/if}
			</button>
			{#if uploading}
				<p in:slide={{ duration: 200 }} class="text-xs" style="color: var(--ink-dim);">
					First time around, the backend hits TMDB for each film (~15s for ~250 ratings). Cached
					after that — same file is instant on re-upload.
				</p>
			{/if}
		</div>

		<div class="md:col-span-2 surface p-5 flex flex-col">
			<div class="flex items-center justify-between">
				<div class="flex items-center gap-2">
					<UsersIcon size={18} style="color: var(--violet);" />
					<h2 class="text-lg font-medium">Group</h2>
					<span class="chip chip-violet">{members.length}</span>
				</div>
				{#if members.length > 0}
					<button
						class="btn btn-ghost btn-pill text-xs px-3 py-1"
						onclick={clearAll}
					>
						<Trash2 size={12} />
						clear
					</button>
				{/if}
			</div>

			<div class="mt-4 flex-1 space-y-2 max-h-72 overflow-auto pr-1">
				{#if members.length === 0}
					<div
						class="border-2 border-dashed rounded-lg p-6 text-center"
						style="border-color: var(--border); color: var(--ink-dim);"
					>
						<FileText size={24} class="mx-auto mb-2 opacity-60" />
						<p class="text-sm">No friends yet.</p>
						<p class="text-xs mt-1" style="color: var(--ink-faint);">
							Add at least 2 to enable group recs.
						</p>
					</div>
				{/if}
				{#each members as m, i (m.hash + i)}
					<div
						class="flex items-center gap-3 px-2 py-2 rounded-lg card-hover surface"
						style="border-radius: 0.6rem;"
						in:fly={{ y: -8, duration: 220 }}
						animate:flip={{ duration: 220 }}
					>
						<Avatar name={m.name} size={36} />
						<div class="flex-1 min-w-0">
							<div class="text-sm font-medium truncate">{m.name}</div>
							<div class="text-[10px] mono truncate" style="color: var(--ink-faint);">
								{m.upload.n_ratings_mapped}/{m.upload.n_ratings_in} mapped · {m.hash.slice(0, 8)}…
							</div>
						</div>
						<button
							class="btn btn-ghost text-xs p-1.5"
							onclick={() => removeMember(i)}
							aria-label="remove"
							style="border-radius: 999px;"
						>
							<Trash2 size={13} />
						</button>
					</div>
				{/each}
			</div>

			<div class="mt-4 space-y-2 pt-2 border-t" style="border-color: var(--border);">
				<button
					class="btn btn-secondary w-full"
					onclick={() => goto('/group/recommendations')}
					disabled={members.length < 2}
					title={members.length < 2 ? 'Need at least 2 members' : ''}
				>
					<Sparkles size={15} />
					Generate group recs
					{#if members.length >= 2}
						<ArrowRight size={14} />
					{/if}
				</button>
				<button
					class="btn btn-ghost w-full text-xs"
					onclick={() => goto('/me')}
				>
					Just me · solo recommendations →
				</button>
			</div>
		</div>
	</div>
</section>
