import Cocoa
import WebKit
import Darwin

class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate, WKNavigationDelegate {
    var window: NSWindow?
    var webView: WKWebView?
    var pythonProcess: Process?
    var port: Int = 8765

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        buildMenu()

        guard let pythonBin = findPythonWithTorch() else {
            showFatalAlert(title: "Python 3 Not Found", message: "Could not find a Python installation with PyTorch.")
            NSApp.terminate(nil)
            return
        }

        guard let serverScript = findServerScript() else {
            showFatalAlert(title: "Server Not Found", message: "Could not locate server.py.")
            NSApp.terminate(nil)
            return
        }

        self.port = findFreePort(start: 8765)
        launchBackend(pythonBin: pythonBin, scriptPath: serverScript, port: self.port)
        createWindow(port: self.port)
    }

    func findPythonWithTorch() -> String? {
        let candidates = [
            "/opt/homebrew/bin/python3",
            "/opt/homebrew/opt/python@3.11/bin/python3.11",
            NSHomeDirectory() + "/.local/bin/python3.11",
            NSHomeDirectory() + "/.local/bin/python3",
            "/usr/local/bin/python3",
            "/usr/bin/python3"
        ]

        for path in candidates {
            if FileManager.default.isExecutableFile(atPath: path) {
                let test = Process()
                test.executableURL = URL(fileURLWithPath: path)
                test.arguments = ["-c", "import torch"]
                test.standardOutput = Pipe()
                test.standardError = Pipe()
                do {
                    try test.run()
                    test.waitUntilExit()
                    if test.terminationStatus == 0 {
                        return path
                    }
                } catch {}
            }
        }
        return nil
    }

    func findServerScript() -> String? {
        if let res = Bundle.main.resourcePath {
            let p = (res as NSString).appendingPathComponent("app/server.py")
            if FileManager.default.fileExists(atPath: p) {
                return p
            }
        }
        return nil
    }

    func findFreePort(start: Int) -> Int {
        for p in start..<(start + 100) {
            var addr = sockaddr_in()
            addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.stride)
            addr.sin_family = sa_family_t(AF_INET)
            addr.sin_port = in_port_t(UInt16(p).bigEndian)
            addr.sin_addr.s_addr = inet_addr("127.0.0.1")

            let sock = socket(AF_INET, SOCK_STREAM, 0)
            var yes: Int32 = 1
            setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.stride))

            let bindRes = withUnsafePointer(to: &addr) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                    Darwin.bind(sock, $0, socklen_t(MemoryLayout<sockaddr_in>.stride))
                }
            }
            Darwin.close(sock)
            if bindRes == 0 {
                return p
            }
        }
        return start
    }

    func launchBackend(pythonBin: String, scriptPath: String, port: Int) {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: pythonBin)
        proc.arguments = [scriptPath, "--no-browser", "--port", "\(port)"]
        if let resURL = Bundle.main.resourceURL {
            proc.currentDirectoryURL = resURL
        }
        
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:" + (env["PATH"] ?? "")
        proc.environment = env

        let devNull = FileHandle.nullDevice
        proc.standardOutput = devNull
        proc.standardError = devNull

        proc.terminationHandler = { [weak self] _ in
            DispatchQueue.main.async {
                self?.window?.close()
                NSApp.terminate(nil)
            }
        }

        do {
            try proc.run()
            self.pythonProcess = proc
        } catch {
            print("Failed to launch python backend: \(error)")
        }
    }

    func createWindow(port: Int) {
        let rect = NSRect(x: 0, y: 0, width: 1280, height: 860)
        let win = NSWindow(
            contentRect: rect,
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )

        win.title = "zar manah diatom tdi"
        win.appearance = NSAppearance(named: .darkAqua)
        win.backgroundColor = NSColor(calibratedRed: 7/255.0, green: 11/255.0, blue: 18/255.0, alpha: 1.0)
        win.minSize = NSSize(width: 960, height: 640)
        win.center()
        win.isReleasedWhenClosed = false
        win.delegate = self

        let config = WKWebViewConfiguration()
        config.preferences.setValue(true, forKey: "developerExtrasEnabled")
        let web = WKWebView(frame: win.contentView!.bounds, configuration: config)
        web.autoresizingMask = [.width, .height]
        web.navigationDelegate = self

        win.contentView = web
        self.window = win
        self.webView = web

        win.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)

        pollAndLoad(attemptsLeft: 50)
    }

    func pollAndLoad(attemptsLeft: Int) {
        let u = URL(string: "http://127.0.0.1:\(self.port)/api/status")!
        let task = URLSession.shared.dataTask(with: u) { [weak self] (_, response, error) in
            guard let self = self else { return }
            if let httpRes = response as? HTTPURLResponse, httpRes.statusCode == 200 {
                DispatchQueue.main.async {
                    let req = URLRequest(url: URL(string: "http://127.0.0.1:\(self.port)")!, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 10.0)
                    self.webView?.load(req)
                }
            } else if attemptsLeft > 0 {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                    self.pollAndLoad(attemptsLeft: attemptsLeft - 1)
                }
            } else {
                DispatchQueue.main.async {
                    let req = URLRequest(url: URL(string: "http://127.0.0.1:\(self.port)")!)
                    self.webView?.load(req)
                }
            }
        }
        task.resume()
    }

    // MARK: - WKNavigationDelegate
    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) { [weak self] in
            guard let self = self else { return }
            let req = URLRequest(url: URL(string: "http://127.0.0.1:\(self.port)")!)
            self.webView?.load(req)
        }
    }

    func buildMenu() {
        let mainMenu = NSMenu()
        let appMenuItem = NSMenuItem()
        mainMenu.addItem(appMenuItem)

        let appMenu = NSMenu()
        let quitItem = NSMenuItem(
            title: "Quit",
            action: #selector(NSApplication.terminate(_:)),
            keyEquivalent: "q"
        )
        appMenu.addItem(quitItem)
        appMenuItem.submenu = appMenu

        NSApp.mainMenu = mainMenu
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        if let win = self.window {
            win.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
        }
        return true
    }

    func windowWillClose(_ notification: Notification) {
        shutdownBackend()
        NSApp.terminate(nil)
    }

    func applicationWillTerminate(_ notification: Notification) {
        shutdownBackend()
    }

    func shutdownBackend() {
        if let proc = self.pythonProcess, proc.isRunning {
            proc.terminate()
            DispatchQueue.global().asyncAfter(deadline: .now() + 0.8) {
                if proc.isRunning {
                    kill(proc.processIdentifier, SIGKILL)
                }
            }
        }
    }

    func showFatalAlert(title: String, message: String) {
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = message
        alert.alertStyle = .critical
        alert.addButton(withTitle: "OK")
        alert.runModal()
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.run()
