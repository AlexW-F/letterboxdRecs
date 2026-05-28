// In-memory + sessionStorage store for uploaded members so navigating
// between pages doesn't drop their hashes.

import type { UploadResult } from './api';

export interface Member {
	name: string;
	hash: string;
	upload: UploadResult;
}

const KEY = 'lbrecs.members';

export function loadMembers(): Member[] {
	if (typeof window === 'undefined') return [];
	try {
		const raw = sessionStorage.getItem(KEY);
		return raw ? (JSON.parse(raw) as Member[]) : [];
	} catch {
		return [];
	}
}

export function saveMembers(members: Member[]): void {
	if (typeof window === 'undefined') return;
	sessionStorage.setItem(KEY, JSON.stringify(members));
}

/**
 * Insert or update a member by hash. Used by upload sites (solo + group join)
 * so /explore and /group/* can find the device's known uploads.
 */
export function addMember(member: Member): void {
	const existing = loadMembers().filter((m) => m.hash !== member.hash);
	existing.push(member);
	saveMembers(existing);
}

export function clearMembers(): void {
	if (typeof window === 'undefined') return;
	sessionStorage.removeItem(KEY);
}
