export default function TorchtabLoading() {
  return (
    <div className="flex items-center justify-center py-20">
      <div className="flex items-center gap-3">
        <div className="w-5 h-5 rounded-full border-2 border-neon-cyan border-t-transparent animate-spin" />
        <span className="text-muted-foreground text-sm">Loading Torch / DDP...</span>
      </div>
    </div>
  );
}
