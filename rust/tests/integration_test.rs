use dev_repo_template_rust::greet;

#[test]
fn greet_end_to_end() {
    assert_eq!(greet("integration"), "Hello, integration!");
}
