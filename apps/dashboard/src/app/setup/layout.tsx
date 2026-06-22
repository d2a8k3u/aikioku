export default function SetupLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen overflow-auto bg-dark-950">
      <div className="mx-auto flex min-h-screen max-w-2xl flex-col justify-center p-6">
        {children}
      </div>
    </div>
  );
}
