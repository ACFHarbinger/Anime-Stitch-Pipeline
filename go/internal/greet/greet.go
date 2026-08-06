// Package greet provides a trivial example function for the go/ module template.
package greet

import "fmt"

// Greet returns a greeting for name.
func Greet(name string) string {
	return fmt.Sprintf("Hello, %s!", name)
}
